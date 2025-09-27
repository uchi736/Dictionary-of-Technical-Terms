#!/usr/bin/env python3
"""
Definition Enricher
===================
専門用語抽出結果に定義を自動付与する統合レイヤー
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector

from dictionary_system.core.models.base_extractor import Term
from dictionary_system.core.rag.bm25_index import BM25Index
from dictionary_system.core.rag.hybrid_search import HybridSearchChain
from dictionary_system.config.rag_config import Config
from dictionary_system.config.prompts import PromptConfig


class DefinitionEnricher:
    """
    専門用語に定義を付与するクラス

    使用方法:
    1. テキストから用語を抽出（SemReRank等）
    2. DefinitionEnricher でテキストをインデックス化
    3. 抽出された用語に定義を自動生成・付与
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        use_pgvector: bool = True,
        use_bm25: bool = True,
        batch_size: int = 5
    ):
        """
        Args:
            config: 設定オブジェクト（Noneの場合は新規作成）
            use_pgvector: PGVectorを使用するか
            use_bm25: BM25を使用するか
            batch_size: バッチ処理のサイズ
        """
        self.config = config or Config()
        self.use_pgvector = use_pgvector and self.config.is_db_configured()
        self.use_bm25 = use_bm25
        self.batch_size = batch_size

        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings() if self.use_pgvector else None
        self.vector_store = None
        self.bm25_index = None
        self.hybrid_chain = None

    def _setup_llm(self) -> AzureChatOpenAI:
        """LLMのセットアップ"""
        return AzureChatOpenAI(
            azure_endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_api_key,
            api_version=self.config.azure_openai_api_version,
            azure_deployment=self.config.azure_openai_chat_deployment_name,
            temperature=0.1,
            streaming=True
        )

    def _setup_embeddings(self) -> AzureOpenAIEmbeddings:
        """埋め込みのセットアップ"""
        return AzureOpenAIEmbeddings(
            azure_endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_api_key,
            api_version=self.config.azure_openai_api_version,
            azure_deployment=self.config.azure_openai_embedding_deployment_name
        )

    def index_text(self, text: str, split_sentences: bool = True):
        """
        テキストをインデックス化

        Args:
            text: インデックス化するテキスト
            split_sentences: 文単位で分割するか
        """
        if split_sentences:
            sentences = [s.strip() + "。" for s in text.split("。") if s.strip()]
        else:
            sentences = [text]

        if self.use_bm25:
            self.bm25_index = BM25Index(use_mecab=True)
            self.bm25_index.add_documents(
                sentences,
                metadatas=[{"sentence_id": i} for i in range(len(sentences))]
            )

        if self.use_pgvector and self.embeddings:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["。", "\n", " "]
            )

            docs = [Document(page_content=sent) for sent in sentences]

            self.vector_store = PGVector.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name="term_definitions",
                connection=self.config.get_db_url(),
                use_jsonb=True
            )

        if self.use_pgvector and self.use_bm25:
            prompt_config = PromptConfig()
            self.hybrid_chain = HybridSearchChain(
                llm=self.llm,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                prompt_config=prompt_config
            )

    def enrich_terms(
        self,
        terms: List[Term],
        top_n: Optional[int] = None,
        verbose: bool = True,
        use_batch: bool = True
    ) -> List[Term]:
        """
        用語リストに定義を付与（LCEL バッチ処理）

        Args:
            terms: 用語のリスト
            top_n: 上位N件のみ処理
            verbose: 進捗表示
            use_batch: バッチ処理を使用（高速）

        Returns:
            定義が付与された用語リスト
        """
        target_terms = terms[:top_n] if top_n else terms

        if use_batch and len(target_terms) > 1:
            if verbose:
                print(f"バッチ処理: {len(target_terms)}件")

            term_texts = [term.term for term in target_terms]
            definitions = self.hybrid_chain.batch(term_texts)

            for term, definition in zip(target_terms, definitions):
                term.definition = definition.strip()

        else:
            for i, term in enumerate(target_terms, 1):
                if verbose:
                    print(f"[{i}/{len(target_terms)}] {term.term}")

                definition = self.hybrid_chain.invoke(term.term)
                term.definition = definition.strip()

        return terms

    async def enrich_terms_async(
        self,
        terms: List[Term],
        top_n: Optional[int] = None,
        verbose: bool = True
    ) -> List[Term]:
        """非同期で用語に定義を付与（LCEL abatch使用）"""
        target_terms = terms[:top_n] if top_n else terms

        if verbose:
            print(f"非同期バッチ処理: {len(target_terms)}件")

        term_texts = [term.term for term in target_terms]
        definitions = await self.hybrid_chain.abatch(term_texts)

        for term, definition in zip(target_terms, definitions):
            term.definition = definition.strip()

        return terms

    def enrich_single_term(self, term: Term) -> Term:
        """単一用語に定義を付与"""
        definition = self.hybrid_chain.invoke(term.term)
        term.definition = definition.strip()
        return term


def enrich_terms_with_definitions(
    terms: List[Term],
    text: str,
    config: Optional[Config] = None,
    top_n: Optional[int] = None,
    verbose: bool = True
) -> List[Term]:
    """
    簡易関数: 用語リストに定義を一括付与

    Args:
        terms: 用語リスト
        text: インデックス化する元テキスト
        config: 設定（Noneの場合は自動生成）
        top_n: 上位N件のみ処理
        verbose: 進捗表示

    Returns:
        定義が付与された用語リスト
    """
    enricher = DefinitionEnricher(config=config)

    if verbose:
        print("テキストをインデックス化中...")

    enricher.index_text(text)

    if verbose:
        print("定義生成開始...\n")

    return enricher.enrich_terms(terms, top_n=top_n, verbose=verbose)