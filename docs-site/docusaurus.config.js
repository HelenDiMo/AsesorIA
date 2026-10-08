// @ts-check
// Configuración del sitio de documentación de AsesorIA.
// Fuente única de contenido: ../docs (el directorio docs/ del repositorio).
// Ver docs-site/README.md (build en Cloudflare Pages).

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AsesorIA',
  tagline: 'Documentación técnica del copiloto fiscal RAG para autónomos en España',
  favicon: 'img/favicon.ico',

  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // URL canónica y baseUrl de GitHub Pages (proyecto, no de usuario):
  // https://HelenDiMo.github.io/AsesorIA/ — despliegue en
  // .github/workflows/deploy-docs.yml (Settings → Pages → GitHub Actions).
  url: 'https://HelenDiMo.github.io',
  baseUrl: '/AsesorIA/',

  onBrokenLinks: 'warn',

  // Los .md de docs/ son Markdown commonMark del repositorio (contienen
  // texto tipo `<...>`, `=`…): sin detección fallaría la compilación MDX.
  markdown: {
    format: 'detect',
  },

  i18n: {
    defaultLocale: 'es',
    locales: ['es'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          // Contenido fuera del directorio del sitio: docs/ de la raíz del repo.
          path: '../docs',
          sidebarPath: './sidebars.js',
          // Sin "edit this page": la fuente vive en docs/ del repositorio.
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/asesoria-social-card.png',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'AsesorIA',
        logo: {
          src: 'img/logo.png',
          alt: 'Logo de AsesorIA',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentación',
          },
          {
            href: 'https://github.com/HelenDiMo/AsesorIA',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentación',
            items: [
              {label: 'Criterios de aceptación', to: '/docs/acceptance_criteria'},
              {label: 'Informe de evaluación (MLflow)', to: '/docs/informe_mlflow'},
              {label: 'Indexación del corpus', to: '/docs/corpus_indexing'},
              {label: 'Despliegue en Render', to: '/docs/deploy_render'},
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AsesorIA. Generado con Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
