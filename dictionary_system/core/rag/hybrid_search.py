#!/usr/bin/env python3
"""
Hybrid Search with LCEL
========================
LangChain Expression Language (LCEL) を使った
ハイブリッド検索 + 定義生成の実装

BM25（キーワード） + ベクトル検索（意味的類似性）
→ RRF融合 → LLM定義生成
"""

import asyncio
from typing import List, Dict, Optional, Any
from collections import defaultdict

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableWithFallbacks
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_postgres import PGVector

from dictionary_system.config.prompts import (
    PromptConfig,
    get_definition_prompt_messages,
    get_similar_terms_prompt_messages
)
from dictionary_system.core.rag.bm25_index import (
    BM25Index,
    reciprocal_rank_fusion
)


class HybridSearchChain:
    """
    LCEL によるハイブリッド検索 + 定義生成チェーン

    アーキテクチャ:
    1. 並列検索 (RunnableParallel)
       - BM25 検索
       - ベクトル検索
       - ローカルコンテキスト検索
    2. RRF 統合
    3. 類似用語抽出
    4. 定義生成 (LLM)
    """

    def __init__(
        self,
        llm: AzureChatOpenAI,
        vector_store: PGVector,
        bm25_index: BM25Index,
        prompt_config: Optional[PromptConfig] = None,
        bm25_top_k: int = 10,
        vector_top_k: int = 10,
        rrf_k: int = 60,
        include_similar_terms: bool = True
    ):
        """
        Args:
            llm: Azure ChatOpenAI インスタンス
            vector_store: PGVector ストア
            bm25_index: BM25 インデックス
            prompt_config: プロンプト設定（デフォルトで新規作成）
            bm25_top_k: BM25 検索結果数
            vector_top_k: ベクトル検索結果数
            rrf_k: RRF パラメータ
            include_similar_terms: 類似用語を含めるか
        """
        self.llm = llm
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.prompt_config = prompt_config or PromptConfig()
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_k = rrf_k
        self.include_similar_terms = include_similar_terms

        self.chain = self._build_chain()

    def _bm25_search(self, inputs: Dict[str, Any]) -> List[Dict]:
        """BM25 検索"""
        term = inputs.get("term", "")
        results = self.bm25_index.search(term, top_k=self.bm25_top_k)

        return [
            {
                "doc_id": doc_id,
                "text": text,
                "score": score,
                "metadata": metadata,
                "source": "bm25"
            }
            for doc_id, score, text, metadata in results
        ]

    def _vector_search(self, inputs: Dict[str, Any]) -> List[Dict]:
        """ベクトル検索"""
        term = inputs.get("term", "")
        results = self.vector_store.similarity_search_with_score(
            term,
            k=self.vector_top_k
        )

        return [
            {
                "doc_id": i,
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
                "source": "vector"
            }
            for i, (doc, score) in enumerate(results)
        ]

    def _local_context_search(self, inputs: Dict[str, Any]) -> List[Dict]:
        """
        ローカルコンテキスト検索
        用語が含まれる文を直接抽出
        """
        term = inputs.get("term", "")
        local_results = []

        for doc in self.bm25_index.documents:
            text = doc["text"]
            if term in text:
                sentences = text.split("。")
                for sentence in sentences:
                    if term in sentence:
                        local_results.append({
                            "doc_id": doc["id"],
                            "text": sentence.strip() + "。",
                            "score": 1.0,
                            "metadata": doc["metadata"],
                            "source": "local"
                        })

        return local_results[:5]

    def _fuse_results(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        RRF で検索結果を統合
        """
        bm25_results = inputs.get("bm25", [])
        vector_results = inputs.get("vector", [])
        local_results = inputs.get("local", [])
        term = inputs.get("term", "")

        bm25_ranking = [(r["doc_id"], r["score"]) for r in bm25_results]
        vector_ranking = [(r["doc_id"], r["score"]) for r in vector_results]
        local_ranking = [(r["doc_id"], r["score"]) for r in local_results]

        fused = reciprocal_rank_fusion(
            [bm25_ranking, vector_ranking, local_ranking],
            k=self.rrf_k
        )

        all_results = {r["doc_id"]: r for r in bm25_results + vector_results + local_results}

        fused_docs = []
        for doc_id, rrf_score in fused[:10]:
            if doc_id in all_results:
                result = all_results[doc_id].copy()
                result["rrf_score"] = rrf_score
                fused_docs.append(result)

        context_text = "\n\n".join([
            f"[{r['source']}] {r['text']}"
            for r in fused_docs[:5]
        ])

        return {
            "term": term,
            "context": context_text,
            "fused_results": fused_docs
        }

    def _extract_similar_terms(self, inputs: Dict[str, Any]) -> str:
        """類似用語を抽出（LLM使用）"""
        if not self.include_similar_terms:
            return "なし"

        term = inputs.get("term", "")
        fused_results = inputs.get("fused_results", [])

        search_results_text = "\n\n".join([
            f"{i+1}. {r['text']}"
            for i, r in enumerate(fused_results[:5])
        ])

        prompt = ChatPromptTemplate.from_messages(
            self.prompt_config.get_similar_terms_prompt()
        )

        chain = prompt | self.llm | StrOutputParser()
        similar_terms = chain.invoke({
            "term": term,
            "search_results": search_results_text
        })

        return similar_terms.strip()

    def _build_chain(self):
        """
        LCEL チェーンを構築

        フロー:
        term → 並列検索 → RRF統合 → 類似用語抽出 → 定義生成
        """
        search_chain = RunnableParallel(
            bm25=RunnableLambda(self._bm25_search),
            vector=RunnableLambda(self._vector_search),
            local=RunnableLambda(self._local_context_search),
            term=RunnablePassthrough()
        )

        fusion_chain = RunnableLambda(self._fuse_results)

        similar_chain = RunnableLambda(
            lambda inputs: {
                **inputs,
                "similar_terms": self._extract_similar_terms(inputs)
            }
        )

        prompt = ChatPromptTemplate.from_messages(
            self.prompt_config.get_definition_prompt(
                use_similar_terms=self.include_similar_terms
            )
        )

        definition_chain = prompt | self.llm | StrOutputParser()

        chain = (
            search_chain
            | fusion_chain
            | similar_chain
            | definition_chain
        )

        return chain

    def invoke(self, term: str) -> str:
        """
        定義生成を実行

        Args:
            term: 専門用語

        Returns:
            生成された定義
        """
        return self.chain.invoke({"term": term})

    async def ainvoke(self, term: str) -> str:
        """非同期で定義生成"""
        return await self.chain.ainvoke({"term": term})

    def stream(self, term: str):
        """ストリーミングで定義生成"""
        return self.chain.stream({"term": term})

    async def astream(self, term: str):
        """非同期ストリーミングで定義生成"""
        async for chunk in self.chain.astream({"term": term}):
            yield chunk

    def batch(self, terms: List[str]) -> List[str]:
        """複数用語の定義を一括生成"""
        inputs = [{"term": term} for term in terms]
        return self.chain.batch(inputs)

    async def abatch(self, terms: List[str]) -> List[str]:
        """非同期バッチ処理"""
        inputs = [{"term": term} for term in terms]
        return await self.chain.abatch(inputs)


class SimpleDefinitionChain:
    """
    シンプル版定義生成チェーン
    類似用語抽出なし、ベクトル検索のみ
    """

    def __init__(
        self,
        llm: AzureChatOpenAI,
        vector_store: PGVector,
        prompt_config: Optional[PromptConfig] = None,
        top_k: int = 5
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.prompt_config = prompt_config or PromptConfig()
        self.top_k = top_k
        self.chain = self._build_chain()

    def _vector_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ベクトル検索のみ"""
        term = inputs.get("term", "")
        results = self.vector_store.similarity_search(term, k=self.top_k)

        context = "\n\n".join([doc.page_content for doc in results])

        return {
            "term": term,
            "context": context
        }

    def _build_chain(self):
        """シンプルなチェーン構築"""
        search_chain = RunnableLambda(self._vector_search)

        prompt = ChatPromptTemplate.from_messages(
            self.prompt_config.get_definition_prompt(use_similar_terms=False)
        )

        chain = (
            {"term": RunnablePassthrough()}
            | search_chain
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def invoke(self, term: str) -> str:
        return self.chain.invoke(term)

    async def ainvoke(self, term: str) -> str:
        return await self.chain.ainvoke(term)


def create_hybrid_chain(
    azure_endpoint: str,
    azure_api_key: str,
    vector_store: PGVector,
    bm25_index: BM25Index,
    deployment_name: str = "gpt-4o",
    **kwargs
) -> HybridSearchChain:
    """
    ハイブリッド検索チェーンのファクトリ関数

    Args:
        azure_endpoint: Azure OpenAI エンドポイント
        azure_api_key: Azure OpenAI API キー
        vector_store: PGVector ストア
        bm25_index: BM25 インデックス
        deployment_name: デプロイメント名
        **kwargs: HybridSearchChain への追加パラメータ

    Returns:
        HybridSearchChain インスタンス
    """
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-02-01",
        azure_deployment=deployment_name,
        temperature=0.1,
        streaming=True
    )

    return HybridSearchChain(
        llm=llm,
        vector_store=vector_store,
        bm25_index=bm25_index,
        **kwargs
    )


def create_simple_chain(
    azure_endpoint: str,
    azure_api_key: str,
    vector_store: PGVector,
    deployment_name: str = "gpt-4o",
    **kwargs
) -> SimpleDefinitionChain:
    """シンプル版チェーンのファクトリ"""
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-02-01",
        azure_deployment=deployment_name,
        temperature=0.1
    )

    return SimpleDefinitionChain(
        llm=llm,
        vector_store=vector_store,
        **kwargs
    )