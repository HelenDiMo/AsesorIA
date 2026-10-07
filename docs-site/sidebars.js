// @ts-check

// Sidebar curado de la documentación de AsesorIA (orden lógico del proyecto).
// Los ids son los nombres de archivo en docs/ sin extensión.

/**
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    'acceptance_criteria',
    'ux-ui-architecture',
    'vector_retrieval',
    'embeddings',
    'chunking_strategy',
    'corpus_cleaning_audit',
    'corpus_indexing',
    'retrieval_evaluation',
    'retrieval_threshold_experiment',
    'retrieval_evidence_review',
    'retrieval_benchmark_audit',
    'retrieval_review_complete',
    'deploy_render',
    'ethics_governance',
  ],
};

export default sidebars;
