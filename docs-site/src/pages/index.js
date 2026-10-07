import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const DOC_LINKS = [
  {to: '/docs/acceptance_criteria', label: 'Criterios de aceptación'},
  {to: '/docs/ux-ui-architecture', label: 'Arquitectura UX/UI'},
  {to: '/docs/vector_retrieval', label: 'Vector store y retriever'},
  {to: '/docs/embeddings', label: 'Capa de embeddings'},
  {to: '/docs/chunking_strategy', label: 'Chunking: tokens y procedencia'},
  {to: '/docs/corpus_cleaning_audit', label: 'Auditoría de limpieza del corpus'},
  {to: '/docs/corpus_indexing', label: 'Indexación del corpus'},
  {to: '/docs/retrieval_evaluation', label: 'Evaluación del retrieval'},
  {to: '/docs/retrieval_threshold_experiment', label: 'Experimento de threshold'},
  {to: '/docs/retrieval_evidence_review', label: 'Revisión de la evidencia'},
  {to: '/docs/retrieval_benchmark_audit', label: 'Auditoría del benchmark'},
  {to: '/docs/retrieval_review_complete', label: 'Revisión cualitativa del benchmark'},
  {to: '/docs/deploy_render', label: 'Despliegue en Render'},
  {to: '/docs/ethics_governance', label: 'Ética y gobernanza'},
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/acceptance_criteria">
            Ver la documentación
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Documentación técnica"
      description="Documentación técnica de AsesorIA: criterios, arquitectura, corpus, retrieval y despliegue.">
      <HomepageHeader />
      <main className={clsx('container', styles.docList)}>
        <Heading as="h2">Índice de documentos</Heading>
        <ul>
          {DOC_LINKS.map((doc) => (
            <li key={doc.to}>
              <Link to={doc.to}>{doc.label}</Link>
            </li>
          ))}
        </ul>
      </main>
    </Layout>
  );
}
