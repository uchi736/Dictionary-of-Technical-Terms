#!/usr/bin/env python3
"""
BM25 Index Implementation for Japanese
=======================================
日本語に対応したBM25インデックス実装
MeCabによる形態素解析 + BM25スコアリング
"""

import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("Warning: MeCab not available. Using simple tokenization.")


class BM25Index:
    """
    BM25による日本語テキスト検索インデックス

    BM25スコア計算式:
    score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

    where:
    - f(qi, D): クエリ語qiの文書D内での出現頻度
    - |D|: 文書Dの長さ（単語数）
    - avgdl: 全文書の平均長
    - k1, b: チューニングパラメータ（通常 k1=1.5, b=0.75）
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_mecab: bool = True
    ):
        """
        Args:
            k1: 語の頻度の飽和を制御（1.2〜2.0推奨）
            b: 文書長の正規化強度（0〜1、0.75推奨）
            use_mecab: MeCabを使用するか（Falseの場合は文字単位）
        """
        self.k1 = k1
        self.b = b
        self.use_mecab = use_mecab and MECAB_AVAILABLE

        if self.use_mecab:
            try:
                self.mecab = MeCab.Tagger("-Owakati")
            except Exception as e:
                print(f"MeCab initialization failed: {e}")
                self.use_mecab = False

        self.documents: List[Dict] = []
        self.doc_freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.corpus_size: int = 0

    def tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化

        Args:
            text: 入力テキスト

        Returns:
            トークンのリスト
        """
        if self.use_mecab:
            result = self.mecab.parse(text).strip()
            tokens = result.split()
        else:
            tokens = list(text.replace(" ", ""))

        return [t for t in tokens if t.strip()]

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        文書をインデックスに追加

        Args:
            texts: 文書テキストのリスト
            metadatas: 各文書のメタデータ（オプション）
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            tokens = self.tokenize(text)
            term_freqs = Counter(tokens)

            self.documents.append({
                "id": len(self.documents),
                "text": text,
                "tokens": tokens,
                "term_freqs": term_freqs,
                "length": len(tokens),
                "metadata": metadata
            })

            for term in term_freqs.keys():
                self.doc_freqs[term] += 1

        self._compute_idf()

    def _compute_idf(self):
        """IDF（逆文書頻度）を計算"""
        self.corpus_size = len(self.documents)
        self.avgdl = sum(doc["length"] for doc in self.documents) / max(self.corpus_size, 1)

        for term, df in self.doc_freqs.items():
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = idf

    def get_scores(self, query: str) -> List[float]:
        """
        クエリに対する全文書のBM25スコアを計算

        Args:
            query: 検索クエリ

        Returns:
            各文書のスコアのリスト（文書IDの順）
        """
        query_tokens = self.tokenize(query)
        query_freqs = Counter(query_tokens)
        scores = []

        for doc in self.documents:
            score = 0.0
            doc_length = doc["length"]
            term_freqs = doc["term_freqs"]

            for term, query_freq in query_freqs.items():
                if term not in term_freqs:
                    continue

                term_freq = term_freqs[term]
                idf = self.idf.get(term, 0.0)

                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avgdl
                )

                score += idf * (numerator / denominator)

            scores.append(score)

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float, str, Dict]]:
        """
        BM25スコアで文書を検索

        Args:
            query: 検索クエリ
            top_k: 返す文書数

        Returns:
            (doc_id, score, text, metadata) のリスト（スコア降順）
        """
        scores = self.get_scores(query)

        results = [
            (i, score, self.documents[i]["text"], self.documents[i]["metadata"])
            for i, score in enumerate(scores)
        ]

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_term_frequencies(self, doc_id: int) -> Dict[str, int]:
        """指定文書の語頻度を取得"""
        if 0 <= doc_id < len(self.documents):
            return dict(self.documents[doc_id]["term_freqs"])
        return {}

    def get_document_length(self, doc_id: int) -> int:
        """指定文書の長さを取得"""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]["length"]
        return 0

    def get_idf(self, term: str) -> float:
        """指定語のIDFを取得"""
        return self.idf.get(term, 0.0)

    def clear(self):
        """インデックスをクリア"""
        self.documents.clear()
        self.doc_freqs.clear()
        self.idf.clear()
        self.avgdl = 0.0
        self.corpus_size = 0


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF)
    複数のランキングを統合

    RRF スコア = Σ 1 / (k + rank_i)

    Args:
        rankings: [(doc_id, score), ...] のリスト（複数のランキング）
        k: 定数（通常60）

    Returns:
        統合されたランキング [(doc_id, rrf_score), ...]
    """
    rrf_scores = defaultdict(float)

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return results


class HybridBM25VectorIndex:
    """
    BM25 + ベクトル検索のハイブリッドインデックス
    RRFで統合
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store,  # LangChain VectorStore
        bm25_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Args:
            bm25_index: BM25インデックス
            vector_store: ベクトルストア（LangChain互換）
            bm25_weight: BM25の重み（0〜1、0.5=同等）
            rrf_k: RRFの定数
        """
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        ハイブリッド検索実行

        Args:
            query: 検索クエリ
            top_k: 返す文書数

        Returns:
            検索結果のリスト [{"text": ..., "score": ..., "metadata": ...}, ...]
        """
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        vector_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)

        bm25_ranking = [(doc_id, score) for doc_id, score, _, _ in bm25_results]
        vector_ranking = [
            (i, score) for i, (doc, score) in enumerate(vector_results)
        ]

        fused_ranking = reciprocal_rank_fusion(
            [bm25_ranking, vector_ranking],
            k=self.rrf_k
        )

        results = []
        for doc_id, rrf_score in fused_ranking[:top_k]:
            if doc_id < len(bm25_results):
                _, bm25_score, text, metadata = bm25_results[doc_id]
                results.append({
                    "text": text,
                    "score": rrf_score,
                    "bm25_score": bm25_score,
                    "metadata": metadata
                })

        return results