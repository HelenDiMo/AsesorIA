# AsesorIA — Sitio de documentación (Docusaurus)

Sitio estático que publica la documentación técnica del repositorio.
**Fuente única**: el directorio `docs/` de la raíz del proyecto (configurado
como `path: '../docs'` en `docusaurus.config.js`); este sitio no duplica
contenido.

## Desarrollo local

```bash
cd docs-site
npm install
npm start          # servidor de desarrollo (http://localhost:3000)
npm run build      # build de producción → build/
npm run serve      # sirve build/ localmente
```

Requisito: Node.js >= 20 (ver `engines` en `package.json`).

## Estructura

| Ruta | Contenido |
|------|-----------|
| `docusaurus.config.js` | título, i18n `es`, `path: '../docs'`, navbar/footer |
| `sidebars.js` | orden curado de los documentos (ids = nombres de `docs/*.md`) |
| `src/pages/index.js` | portada con índice enlazado a todos los documentos |
| `../docs/*.md` | documentación (editar SIEMPRE aquí) |

Al añadir un documento nuevo: crear `docs/<archivo>.md` con un `# Título` y
añadir su id a `sidebars.js` (y, si se quiere en la portada, a `DOC_LINKS`
en `src/pages/index.js`).

## Despliegue en Cloudflare Pages

Configuración del proyecto de Pages:

| Opción | Valor |
|--------|-------|
| Root directory | `docs-site` |
| Build command | `npm run build` |
| Build output directory | `build` |
| Node version | 20 (o superior) |

Antes del primer despliegue, ajustar `url` en `docusaurus.config.js` al
dominio asignado por Cloudflare (afecta a sitemap y enlaces canónicos).

En CI (`.github/workflows/ci.yml`, job `docs`) el build se verifica en cada
push/PR para impedir que la documentación deje de compilar.

## Decisiones y límites

- Blog deshabilitado (`blog: false`): la documentación es el único contenido.
- `onBrokenLinks: 'warn'`: los documentos contienen enlaces relativos al
  repositorio (p. ej. `ui/README.md`) que no son rutas del sitio; se revisan
  los avisos del build en lugar de romperlo.
- Los PDF de `data/` y el corpus indexado NO se publican aquí: solo Markdown
  de `docs/`.
