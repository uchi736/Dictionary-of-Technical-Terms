#!/usr/bin/env python3
"""
Term Extraction Pipeline with LCEL
===================================
LangChain LCEL を使った専門用語抽出 + 定義生成パイプライン
"""

from typing import List, Dict, Optional, Any
from pathlib import Path

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableWithFallbacks,
    Runnable
)

from dictionary_system.core.models.base_extractor import Term
from dictionary_system.core.extractors.semrerank_correct import SemReRankExtractor
from dictionary_system.core.rag.definition_enricher import DefinitionEnricher
from dictionary_system.config.rag_config import Config


class ExtractionPipeline:
    """
    LCEL による専門用語抽出 + 定義生成パイプライン

    パイプライン構成:
    text → 前処理 → 用語抽出 (SemReRank) → 定義付与 (RAG) → 後処理 → 結果
    """

    def __init__(
        self,
        extractor: Optional[SemReRankExtractor] = None,
        enricher: Optional[DefinitionEnricher] = None,
        config: Optional[Config] = None,
        enable_definitions: bool = True,
        top_n_terms: int = 10
    ):
        """
        Args:
            extractor: 用語抽出器（Noneの場合はデフォルト作成）
            enricher: 定義付与器（Noneの場合はデフォルト作成）
            config: 設定
            enable_definitions: 定義生成を有効にするか
            top_n_terms: 上位N件の用語を処理
        """
        self.config = config or Config()
        self.enable_definitions = enable_definitions
        self.top_n_terms = top_n_terms

        self.extractor = extractor or self._create_default_extractor()
        self.enricher = enricher or DefinitionEnricher(config=self.config)

        self.pipeline = self._build_pipeline()

    def _create_default_extractor(self) -> SemReRankExtractor:
        """デフォルトの抽出器を作成"""
        return SemReRankExtractor(
            base_ate_method="tfidf",
            use_azure_embeddings=True,
            auto_select_seeds=True,
            seed_z=50,
            use_elbow_detection=True,
            min_seed_count=5,
            max_seed_ratio=0.7
        )

    def _preprocess_text(self, text: str) -> str:
        """前処理"""
        return text.strip()

    def _extract_terms(self, text: str) -> List[Term]:
        """用語抽出"""
        return self.extractor.extract(text)

    def _index_text_for_rag(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """RAG用にテキストをインデックス化"""
        text = inputs.get("text", "")
        terms = inputs.get("terms", [])

        if self.enable_definitions and len(terms) > 0:
            self.enricher.index_text(text)

        return {
            "text": text,
            "terms": terms,
            "enricher": self.enricher
        }

    def _enrich_definitions(self, inputs: Dict[str, Any]) -> List[Term]:
        """定義を付与"""
        terms = inputs.get("terms", [])

        if not self.enable_definitions or len(terms) == 0:
            return terms

        enricher = inputs.get("enricher")
        enriched = enricher.enrich_terms(
            terms,
            top_n=self.top_n_terms,
            verbose=False,
            use_batch=True
        )

        return enriched

    def _postprocess(self, terms: List[Term]) -> Dict[str, Any]:
        """後処理と結果整形"""
        return {
            "terms": terms,
            "count": len(terms),
            "top_terms": terms[:self.top_n_terms] if terms else []
        }

    def _build_pipeline(self) -> Runnable:
        """
        LCEL パイプラインを構築

        フロー:
        text → 前処理 → 用語抽出 → [並列: テキスト保持, インデックス化] → 定義付与 → 後処理
        """
        preprocess_chain = RunnableLambda(self._preprocess_text)

        extract_chain = RunnableLambda(self._extract_terms)

        parallel_chain = RunnableParallel(
            text=RunnablePassthrough(),
            terms=RunnablePassthrough(),
            enricher=RunnablePassthrough()
        )

        index_chain = RunnableLambda(self._index_text_for_rag)

        enrich_chain = RunnableLambda(self._enrich_definitions)

        postprocess_chain = RunnableLambda(self._postprocess)

        pipeline = (
            preprocess_chain
            | RunnableLambda(lambda text: {"text": text, "terms": self._extract_terms(text)})
            | index_chain
            | enrich_chain
            | postprocess_chain
        )

        return pipeline

    def invoke(self, text: str) -> Dict[str, Any]:
        """
        パイプライン実行

        Args:
            text: 入力テキスト

        Returns:
            {"terms": [...], "count": N, "top_terms": [...]}
        """
        return self.pipeline.invoke(text)

    async def ainvoke(self, text: str) -> Dict[str, Any]:
        """非同期実行"""
        return await self.pipeline.ainvoke(text)

    def batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """バッチ処理"""
        return self.pipeline.batch(texts)

    async def abatch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """非同期バッチ処理"""
        return await self.pipeline.abatch(texts)

    def stream(self, text: str):
        """ストリーミング（部分結果を返す）"""
        for chunk in self.pipeline.stream(text):
            yield chunk


class SimplifiedExtractionPipeline:
    """
    シンプル版パイプライン（定義生成なし）
    """

    def __init__(
        self,
        extractor: Optional[SemReRankExtractor] = None,
        top_n_terms: int = 10
    ):
        self.extractor = extractor or SemReRankExtractor(
            base_ate_method="tfidf",
            use_azure_embeddings=True,
            auto_select_seeds=True
        )
        self.top_n_terms = top_n_terms
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Runnable:
        """シンプルなパイプライン"""
        preprocess = RunnableLambda(lambda t: t.strip())
        extract = RunnableLambda(lambda t: self.extractor.extract(t))
        postprocess = RunnableLambda(
            lambda terms: {
                "terms": terms,
                "count": len(terms),
                "top_terms": terms[:self.top_n_terms]
            }
        )

        return preprocess | extract | postprocess

    def invoke(self, text: str) -> Dict[str, Any]:
        return self.pipeline.invoke(text)


def create_extraction_pipeline(
    enable_definitions: bool = True,
    config: Optional[Config] = None,
    **kwargs
) -> ExtractionPipeline:
    """
    抽出パイプラインのファクトリ関数

    Args:
        enable_definitions: 定義生成を有効にするか
        config: 設定オブジェクト
        **kwargs: ExtractionPipeline への追加パラメータ

    Returns:
        ExtractionPipeline インスタンス
    """
    return ExtractionPipeline(
        config=config,
        enable_definitions=enable_definitions,
        **kwargs
    )


def create_simple_pipeline(
    config: Optional[Config] = None,
    **kwargs
) -> SimplifiedExtractionPipeline:
    """シンプル版パイプラインのファクトリ"""
    return SimplifiedExtractionPipeline(**kwargs)