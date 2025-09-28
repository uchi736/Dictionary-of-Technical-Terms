"""
RAG (Retrieval-Augmented Generation) Module
============================================
ハイブリッド検索 + LLM による定義生成

主要コンポーネント:
- BM25Index: 日本語対応BM25検索
- HybridSearchChain: LCEL による BM25 + ベクトル検索 + 定義生成
- SimpleDefinitionChain: シンプル版（ベクトル検索のみ）
"""

from dictionary_system.core.rag.bm25_index import (
    BM25Index,
    reciprocal_rank_fusion,
    HybridBM25VectorIndex
)

from dictionary_system.core.rag.hybrid_search import (
    HybridSearchChain,
    SimpleDefinitionChain,
    create_hybrid_chain,
    create_simple_chain
)

from dictionary_system.core.rag.definition_enricher import (
    DefinitionEnricher,
    enrich_terms_with_definitions,
    filter_technical_terms_by_definition
)

from dictionary_system.core.rag.extraction_pipeline import (
    ExtractionPipeline,
    SimplifiedExtractionPipeline,
    create_extraction_pipeline,
    create_simple_pipeline
)

from dictionary_system.core.rag.synonym_extractor import (
    SynonymExtractor,
    SynonymPair,
    extract_synonym_groups,
    HierarchicalSynonymExtractor,
    SynonymHierarchy,
    extract_synonym_hierarchy
)

__all__ = [
    "BM25Index",
    "reciprocal_rank_fusion",
    "HybridBM25VectorIndex",
    "HybridSearchChain",
    "SimpleDefinitionChain",
    "create_hybrid_chain",
    "create_simple_chain",
    "DefinitionEnricher",
    "enrich_terms_with_definitions",
    "filter_technical_terms_by_definition",
    "ExtractionPipeline",
    "SimplifiedExtractionPipeline",
    "create_extraction_pipeline",
    "create_simple_pipeline",
    "SynonymExtractor",
    "SynonymPair",
    "extract_synonym_groups",
    "HierarchicalSynonymExtractor",
    "SynonymHierarchy",
    "extract_synonym_hierarchy"
]