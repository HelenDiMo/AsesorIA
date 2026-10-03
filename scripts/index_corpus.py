import csv
import json
from datetime import datetime, timezone
from hashlib import sha256

from src.common.schemas import ChunkMetadata
from src.common.tokenization import load_tokenization
from src.ingestion.chunking import (
    ChunkingConfig, chunk_document, join_pages_with_spans, get_page_range,
)

from src.common.settings import PROJECT_ROOT
from src.ingestion.loaders import load_document, LoadedPage
from src.ingestion.cleaning import clean_text, remove_repeated_lines


def check_corpus():
    """Comprueba que cada PDF del catálogo existe y no está vacío."""
    catalog_path = PROJECT_ROOT / "data" / "sources.csv"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    with catalog_path.open(encoding="utf-8-sig", newline="") as file:
        sources = list(csv.DictReader(file))

    if not sources:
        raise ValueError("El catálogo de documentos está vacío")

    missing = []

    for source in sources:
        path = raw_dir / source["filename"]

        if not path.is_file() or path.stat().st_size == 0:
            missing.append(source["filename"])
            print(f"FALTA O ESTÁ VACÍO: {source['filename']}")
        else:
            print(f"OK: {source['doc_id']} → {source['filename']}")

    if missing:
        raise FileNotFoundError(
            f"Hay {len(missing)} documentos ausentes o vacíos"
        )

    print(f"\nCatálogo comprobado: {len(sources)} documentos.")
    return sources


def load_clean_pages(source):
    """Carga un documento del catálogo y conserva la numeración de sus páginas."""
    path = PROJECT_ROOT / "data" / "raw" / source["filename"]
    pages = load_document(path)

    # Política auditada para los PDF actuales; revisar si cambia el corpus.
    # En los manuales AEAT conservamos las viñetas: solo clean_text.
    if source["tax"] == "RETA":
        pages = remove_repeated_lines(pages)

    return [
        LoadedPage(text=clean_text(page.text), page=page.page)
        for page in pages
    ]


# Candidatas experimentales; ninguna se considera ganadora todavía.
CONFIGURATIONS = ((256, 32), (384, 48), (480, 64))


def validate_chunks(chunks, pages, metadata, config, tokenization):
    """Comprueba límites, procedencia y cobertura sin generar vectores."""
    text, spans = join_pages_with_spans(pages)
    covered_until = 0
    previous_start = -1
    counts = []
    for index, chunk in enumerate(chunks):
        def require(condition, message):
            if not condition:
                raise ValueError(f"{metadata.doc_id}, chunk {index}: {message}")

        require(0 <= chunk.start < chunk.end <= len(text), "offsets inválidos")
        require(chunk.start >= previous_start, "fragmentos desordenados")
        require(chunk.text == text[chunk.start:chunk.end], "texto original alterado")
        require(bool(chunk.text.strip()), "fragmento vacío")
        require(not text[covered_until:chunk.start].strip(), "texto sin cubrir")
        expected_pages = get_page_range(chunk.start, chunk.end, spans)
        require((chunk.metadata.page, chunk.metadata.page_end) == expected_pages,
                "rango de páginas incorrecto")
        expected_metadata = metadata.model_dump()
        expected_metadata.update(page=expected_pages[0], page_end=expected_pages[1],
                                 chunk_index=index)
        require(chunk.metadata.model_dump() == expected_metadata, "metadata alterada")
        expected_embedding_text = tokenization.prepare(
            chunk.text, chunk.metadata.section_label, chunk.metadata.section_path
        )
        require(chunk.embedding_text == expected_embedding_text, "contexto incorrecto")
        count = tokenization.count(chunk.embedding_text)
        require(count == chunk.token_count, "conteo de tokens incorrecto")
        require(count <= min(config.max_tokens, tokenization.max_input_tokens),
                "presupuesto de tokens superado")
        require(0 <= chunk.overlap_tokens <= config.overlap_tokens,
                "overlap fuera del presupuesto")
        counts.append(count)
        covered_until = max(covered_until, chunk.end)
        previous_start = chunk.start

    if text[covered_until:].strip():
        raise ValueError(f"{metadata.doc_id}: texto final sin cubrir")
    return {
        "chunks": len(chunks),
        "min_tokens": min(counts, default=0),
        "max_tokens": max(counts, default=0),
        "max_overlap_tokens": max((c.overlap_tokens for c in chunks), default=0),
        "cross_page_chunks": sum(c.metadata.page != c.metadata.page_end for c in chunks),
    }


def main():
    sources = check_corpus()
    tokenization = load_tokenization()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenization.tokenizer.name_or_path,
        "max_input_tokens": tokenization.max_input_tokens,
        "passage_prefix": tokenization.passage_prefix,
        "cleaning": "RETA: remove_repeated_lines + clean_text; AEAT: clean_text",
        "sections": "No inferred headings; shared chunking defaults",
        "documents": [],
        "totals": {f"{size}/{overlap}": 0 for size, overlap in CONFIGURATIONS},
    }
    for source in sources:
        print(f"\nLeyendo {source['doc_id']}...", flush=True)
        pages = load_clean_pages(source)
        metadata = ChunkMetadata(
            **{key: value for key, value in source.items() if key != "filename"},
            source_scope="public",
        )
        path = PROJECT_ROOT / "data" / "raw" / source["filename"]
        document = {
            "doc_id": metadata.doc_id,
            "filename": source["filename"],
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "pages": len(pages),
            "empty_pages": sum(not page.text.strip() for page in pages),
            "configurations": {},
        }
        print(f"Páginas: {len(pages)} | Vacías: {document['empty_pages']}", flush=True)
        for size, overlap in CONFIGURATIONS:
            key = f"{size}/{overlap}"
            print(f"  Dividiendo y validando {key}...", flush=True)
            config = ChunkingConfig(size, overlap)
            chunks = chunk_document(pages, metadata, config=config,
                                    tokenization=tokenization)
            stats = validate_chunks(chunks, pages, metadata, config, tokenization)
            document["configurations"][key] = stats
            report["totals"][key] += stats["chunks"]
            print(f"  {key}: {stats}", flush=True)
            del chunks
        report["documents"].append(document)

    # Solo un informe JSON local: no crea colecciones ni escribe vectores.
    output = PROJECT_ROOT / "chroma_db" / "corpus_chunking_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotales de chunks: {report['totals']}")
    print(f"Informe: {output}")
    print("Validación técnica; no mide calidad del retrieval ni elige parámetros óptimos.")


if __name__ == "__main__":
    main()
