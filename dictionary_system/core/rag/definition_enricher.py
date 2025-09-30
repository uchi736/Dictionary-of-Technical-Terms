#!/usr/bin/env python3
"""
Definition Enricher
===================
専門用語抽出結果に定義を自動付与する統合レイヤー
"""

import os
import json
from typing import List, Optional, Dict
from pathlib import Path

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dictionary_system.core.models.base_extractor import Term
from dictionary_system.core.rag.bm25_index import BM25Index
from dictionary_system.core.rag.hybrid_search import HybridSearchChain
from dictionary_system.config.rag_config import Config
from dictionary_system.config.prompts import PromptConfig, get_technical_term_judgment_prompt_messages


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
            self.bm25_index = BM25Index()
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
        use_batch: bool = True,
        hierarchy: Optional[Dict] = None
    ) -> List[Term]:
        """
        用語リストに定義を付与（LCEL バッチ処理）

        Args:
            terms: 用語のリスト
            top_n: 上位N件のみ処理
            verbose: 進捗表示
            use_batch: バッチ処理を使用（高速）
            hierarchy: クラスタ階層情報（SynonymHierarchy辞書）

        Returns:
            定義が付与された用語リスト
        """
        target_terms = terms[:top_n] if top_n else terms

        # 階層情報がある場合は階層対応チェインを使用
        if hierarchy:
            return self._enrich_terms_with_hierarchy(target_terms, hierarchy, verbose, use_batch)

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

    def _build_hierarchy_context(self, term: str, hierarchy: Dict) -> str:
        """
        用語の階層的コンテキストを構築

        Args:
            term: 対象用語
            hierarchy: クラスタ階層情報（{代表語: SynonymHierarchy}）

        Returns:
            階層的コンテキスト文字列
        """
        # 用語が所属するクラスタを探す
        parent_cluster = None
        sibling_terms = []
        cluster_name = None

        for rep, node in hierarchy.items():
            if term in node.terms:
                cluster_name = node.category_name or f"クラスタ {node.cluster_id}"
                sibling_terms = [t for t in node.terms if t != term]

                # 親クラスタ情報（将来的に実装可能）
                # HDBSCANのcondensed_treeから親クラスタ取得

                break

        if not cluster_name:
            return "階層情報なし"

        context_parts = [f"【所属カテゴリ】{cluster_name}"]

        if sibling_terms:
            # 同じクラスタの関連用語（最大5件）
            sibling_list = ", ".join(sibling_terms[:5])
            if len(sibling_terms) > 5:
                sibling_list += f" など（他{len(sibling_terms)-5}件）"
            context_parts.append(f"【関連用語（同カテゴリ）】{sibling_list}")

        return "\n".join(context_parts)

    def _enrich_terms_with_hierarchy(
        self,
        terms: List[Term],
        hierarchy: Dict,
        verbose: bool = True,
        use_batch: bool = True
    ) -> List[Term]:
        """
        階層情報を考慮した定義生成

        Args:
            terms: 用語リスト
            hierarchy: クラスタ階層情報
            verbose: 進捗表示
            use_batch: バッチ処理使用

        Returns:
            定義が付与された用語リスト
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableParallel, RunnablePassthrough
        from dictionary_system.config.prompts import get_definition_prompt_messages_with_hierarchy

        if verbose:
            print(f"階層対応定義生成: {len(terms)}件")

        # 階層対応プロンプト
        prompt = ChatPromptTemplate.from_messages(
            get_definition_prompt_messages_with_hierarchy()
        )

        # 階層対応チェイン構築
        hierarchy_chain = (
            RunnableParallel({
                "term": RunnablePassthrough(),
                "context": lambda x: self._get_context_for_term(x),
                "hierarchy_context": lambda x: self._build_hierarchy_context(x, hierarchy)
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

        if use_batch and len(terms) > 1:
            term_texts = [term.term for term in terms]
            definitions = hierarchy_chain.batch(term_texts)

            for term, definition in zip(terms, definitions):
                term.definition = definition.strip()
        else:
            for i, term in enumerate(terms, 1):
                if verbose:
                    print(f"[{i}/{len(terms)}] {term.term}")

                definition = hierarchy_chain.invoke(term.term)
                term.definition = definition.strip()

        return terms

    def _get_context_for_term(self, term: str) -> str:
        """用語の関連コンテキストを取得"""
        if self.use_pgvector and self.vector_store:
            results = self.vector_store.similarity_search(term, k=3)
            return "\n".join([doc.page_content for doc in results])
        elif self.use_bm25 and self.bm25_index:
            from rank_bm25 import BM25Okapi
            query_tokens = term.split()
            scores = self.bm25_index.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            return "\n".join([self.bm25_index.corpus[i] for i in top_indices if i < len(self.bm25_index.corpus)])
        return ""


def enrich_terms_with_definitions(
    terms: List[Term],
    text: str,
    config: Optional[Config] = None,
    top_n: Optional[int] = None,
    verbose: bool = True,
    hierarchy: Optional[Dict] = None
) -> List[Term]:
    """
    簡易関数: 用語リストに定義を一括付与

    Args:
        terms: 用語リスト
        text: インデックス化する元テキスト
        config: 設定（Noneの場合は自動生成）
        top_n: 上位N件のみ処理
        verbose: 進捗表示
        hierarchy: クラスタ階層情報（SynonymHierarchy辞書）

    Returns:
        定義が付与された用語リスト
    """
    enricher = DefinitionEnricher(config=config)

    if verbose:
        print("テキストをインデックス化中...")

    enricher.index_text(text)

    if verbose:
        print("定義生成開始...\n")

    return enricher.enrich_terms(terms, top_n=top_n, verbose=verbose, hierarchy=hierarchy)


def filter_technical_terms_by_definition(
    terms: List[Term],
    config: Optional[Config] = None,
    llm_model: str = "gpt-4.1-mini",
    verbose: bool = True
) -> List[Term]:
    """
    定義ベースで専門用語をフィルタリング（バッチ処理版）

    Args:
        terms: 定義付き用語リスト
        config: 設定
        llm_model: LLMモデル名
        verbose: 進捗表示

    Returns:
        専門用語と判定された用語のみのリスト
    """
    config = config or Config()

    terms_with_def = [t for t in terms if t.definition]

    if not terms_with_def:
        return []

    if verbose:
        print(f"専門用語判定: {len(terms_with_def)}件をバッチ処理中...")

    llm = AzureChatOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        azure_deployment=llm_model,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages(
        get_technical_term_judgment_prompt_messages()
    )
    chain = prompt | llm | StrOutputParser()

    batch_inputs = [
        {"term": term.term, "definition": term.definition}
        for term in terms_with_def
    ]

    result_texts = chain.batch(batch_inputs)

    technical_terms = []
    for term, result_text in zip(terms_with_def, result_texts):
        result = _parse_judgment_result(result_text)

        if result and result.get("is_technical", False):
            technical_terms.append(term)
            if verbose:
                print(f"  [OK] {term.term}: 専門用語 (信頼度: {result.get('confidence', 0):.2f})")
        elif verbose:
            print(f"  [NG] {term.term}: 一般用語")

    if verbose:
        print(f"\n専門用語: {len(technical_terms)}/{len(terms_with_def)}件")

    return technical_terms


def _parse_judgment_result(text: str) -> Optional[Dict]:
    """LLM判定結果のJSONをパース"""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None