#!/usr/bin/env python3
"""
SemRe-Rank 正式実装
===================
論文: "SemRe-Rank: Improving Automatic Term Extraction By Incorporating
Semantic Relatedness With Personalised PageRank" (Zhang et al., 2017)

主要な特徴:
1. シード用語による Personalized PageRank
2. 文書レベルグラフの構築と集約
3. 意味的関連性閾値（relmin=0.5, reltop=15%）
4. 基底ATEスコアと意味的重要度の統合
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import re
import math
import os
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from dictionary_system.core.models.base_extractor import BaseExtractor, Term

# Azure OpenAI Embeddings
try:
    from langchain_openai import AzureOpenAIEmbeddings
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Warning: langchain_openai not available. Using SentenceTransformer.")

logger = logging.getLogger(__name__)

class SemReRank:
    """
    SemRe-Rank: 意味的関連性とPersonalized PageRankによるATE改善
    """

    def __init__(
        self,
        use_azure_embeddings: bool = True,  # Azure埋め込みを優先
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",  # フォールバック
        relmin: float = 0.5,        # 最小関連性閾値
        reltop: float = 0.15,       # 上位15%の関連語を選択
        alpha: float = 0.85,        # PageRankダンピングファクタ
        seed_z: int = 100,          # シード選定用の上位候補数
        manual_seeds: Optional[List[str]] = None,  # 手動指定のシード用語
        auto_select_seeds: bool = False,  # 自動シード選択（無監督版）
        use_elbow_detection: bool = True,  # エルボー法を使用
        min_seed_count: int = 10,   # 最小シード数
        max_seed_ratio: float = 0.7  # 最大シード割合（seed_zの70%まで）
    ):
        """
        Args:
            use_azure_embeddings: Azure OpenAI埋め込みを使用するか
            embedding_model: 埋め込みモデル名（Azure非使用時）
            relmin: 最小意味的関連性閾値（0.5推奨）
            reltop: 関連語選択の上位パーセンテージ（0.15=15%推奨）
            alpha: PageRankのダンピングファクタ（0.85推奨）
            seed_z: シード候補として選ぶ上位用語数（100または200）
            manual_seeds: 手動で指定したシード用語リスト
            auto_select_seeds: Trueの場合、上位z個を自動的にシードとする
        """
        self.use_azure_embeddings = use_azure_embeddings and AZURE_AVAILABLE

        # Azure埋め込み設定
        if self.use_azure_embeddings:
            try:
                self.azure_embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    azure_deployment="text-embedding-3-small"  # .envの設定に合わせる
                )
                logger.info("Using Azure OpenAI Embeddings (text-embedding-3-small)")
            except Exception as e:
                logger.warning(f"Azure Embeddings initialization failed: {e}")
                self.use_azure_embeddings = False
                self.embedder = SentenceTransformer(embedding_model)
                logger.info(f"Falling back to SentenceTransformer: {embedding_model}")
        else:
            self.embedder = SentenceTransformer(embedding_model)
            logger.info(f"Using SentenceTransformer: {embedding_model}")

        self.relmin = relmin
        self.reltop = reltop
        self.alpha = alpha
        self.seed_z = seed_z
        self.manual_seeds = manual_seeds or []
        self.auto_select_seeds = auto_select_seeds
        self.use_elbow_detection = use_elbow_detection
        self.min_seed_count = min_seed_count
        self.max_seed_ratio = max_seed_ratio

    def enhance_ate_scores(
        self,
        text: str,
        candidate_terms: Dict[str, float],  # {term: base_ate_score}（Stage B用）
        term_frequencies: Optional[Dict[str, int]] = None,
        seed_selection_scores: Optional[Dict[str, float]] = None  # Stage A用スコア
    ) -> Dict[str, float]:
        """
        SemRe-Rankによるスコア改善のメインメソッド

        Args:
            text: 対象文書のテキスト
            candidate_terms: 候補用語とその基底ATEスコア（Stage B: 最終ランク付け用）
            term_frequencies: 用語の頻度（シード選択用）
            seed_selection_scores: シード選択用スコア（Stage A: C-value重視）

        Returns:
            改善されたスコア辞書 {term: enhanced_score}
        """
        if not candidate_terms:
            return {}

        # シード選択にはStage Aのスコアを使用（指定された場合）
        scores_for_seed_selection = seed_selection_scores if seed_selection_scores else candidate_terms

        # 1. シード用語を選定（Stage Aスコア使用）
        seed_terms = self._select_seed_terms(scores_for_seed_selection, term_frequencies)
        logger.info(f"Selected {len(seed_terms)} seed terms")

        # 2. 候補用語から単語を抽出
        all_words = self._extract_words_from_terms(candidate_terms.keys())

        # 3. 単語の埋め込みを計算
        word_embeddings = self._compute_word_embeddings(list(all_words))

        # 4. ペアワイズ意味的関連性を計算
        relatedness = self._compute_pairwise_relatedness(all_words, word_embeddings)

        # 5. 文書を文に分割
        documents = self._split_into_sentences(text)

        # 6. 各文書（文）でグラフを構築しPageRankを実行
        word_importance = defaultdict(float)

        for doc in documents:
            # 文書に含まれる単語を抽出
            doc_words = self._extract_words_from_text(doc) & all_words
            if len(doc_words) < 2:
                continue

            # グラフ構築
            graph = self._build_semantic_graph(doc_words, relatedness)
            if len(graph) == 0:
                continue

            # Personalized PageRank実行
            pagerank_scores = self._personalized_pagerank(graph, doc_words, seed_terms)

            # 文書レベルのスコアを集約
            for word, score in pagerank_scores.items():
                word_importance[word] += score

        # 7. 候補用語のスコアを改訂（Stage Bスコア使用）
        enhanced_scores = self._revise_scores(
            candidate_terms,
            word_importance,
            all_words
        )

        return enhanced_scores

    def _select_seed_terms(
        self,
        candidate_terms: Dict[str, float],
        term_frequencies: Optional[Dict[str, int]] = None
    ) -> Set[str]:
        """
        シード用語を選定（論文3.2.2節 + エルボー法改良）

        Args:
            candidate_terms: 候補用語と基底スコア
            term_frequencies: 用語頻度

        Returns:
            シード用語の集合
        """
        # 手動シードが指定されている場合
        if self.manual_seeds:
            return set(self.manual_seeds)

        # 頻度順でソート（頻度情報がない場合はスコア順）
        if term_frequencies:
            sorted_terms = sorted(
                candidate_terms.keys(),
                key=lambda x: term_frequencies.get(x, 0),
                reverse=True
            )
            scores = [term_frequencies.get(t, 0) for t in sorted_terms]
        else:
            sorted_terms = sorted(
                candidate_terms.keys(),
                key=lambda x: candidate_terms[x],
                reverse=True
            )
            scores = [candidate_terms[t] for t in sorted_terms]

        # 上位z個に制限
        top_z = sorted_terms[:min(self.seed_z, len(sorted_terms))]
        top_scores = scores[:min(self.seed_z, len(scores))]

        if self.auto_select_seeds:
            if self.use_elbow_detection and len(top_z) > 3:
                # エルボー法で自動決定
                seed_count = self._detect_elbow_point(top_scores)

                # 最小・最大制約を適用
                seed_count = max(self.min_seed_count, seed_count)
                max_seeds = int(self.seed_z * self.max_seed_ratio)
                seed_count = min(seed_count, max_seeds, len(top_z))

                logger.info(f"Elbow detected at position {seed_count} (min={self.min_seed_count}, max={max_seeds})")
                return set(top_z[:seed_count])
            else:
                # エルボー法を使わない場合は上位50%
                seed_count = max(self.min_seed_count, len(top_z)//2)
                logger.info(f"Auto-selected top {seed_count} terms as seeds")
                return set(top_z[:seed_count])
        else:
            # 監督版：デモのため上位50%を仮選択
            logger.warning("Manual seed verification required. Using top 50% as demo")
            return set(top_z[:len(top_z)//2])

    def _detect_elbow_point(self, scores: List[float]) -> int:
        """
        エルボー法でスコアの急落点を検出

        Args:
            scores: ソート済みスコアリスト（降順）

        Returns:
            エルボーポイントのインデックス
        """
        if len(scores) < 3:
            return len(scores)

        # スコアの差分を計算
        diffs = []
        for i in range(1, len(scores)):
            diff = scores[i-1] - scores[i]
            diffs.append(diff)

        # 2階差分で変化率の変化を検出
        if len(diffs) > 1:
            second_diffs = []
            for i in range(1, len(diffs)):
                second_diff = abs(diffs[i] - diffs[i-1])
                second_diffs.append(second_diff)

            # 最大変化点を検出
            if second_diffs:
                elbow_idx = np.argmax(second_diffs) + 2
                return elbow_idx

        # フォールバック：差分が最大の位置
        if diffs:
            return np.argmax(diffs) + 1

        return len(scores) // 2

    def _extract_words_from_terms(self, terms: List[str]) -> Set[str]:
        """候補用語から単語を抽出"""
        words = set()
        for term in terms:
            # 単純な分割（実際は形態素解析を使用）
            # 日本語対応のため、複数の方法を試す
            parts = []

            # スペース分割
            parts.extend(term.split())

            # 基本的な日本語分割（簡易版）
            # カタカナ、ひらがな、漢字の境界で分割
            pattern = r'[ァ-ヶー]+|[ぁ-ゔー]+|[一-龯]+|[A-Za-z0-9]+'
            parts.extend(re.findall(pattern, term))

            for word in parts:
                if len(word) > 1:  # 1文字の単語は除外
                    words.add(word)

        return words

    def _extract_words_from_text(self, text: str) -> Set[str]:
        """テキストから単語を抽出"""
        words = set()
        # 基本的な単語抽出（形態素解析器を使うべき）
        pattern = r'[ァ-ヶー]+|[ぁ-ゔー]+|[一-龯]+|[A-Za-z0-9]+'
        for match in re.finditer(pattern, text):
            word = match.group()
            if len(word) > 1:
                words.add(word)
        return words

    def _compute_word_embeddings(self, words: List[str]) -> Dict[str, np.ndarray]:
        """単語の埋め込みベクトルを計算"""
        if self.use_azure_embeddings:
            embeddings_list = self.azure_embeddings.embed_documents(words)
            embeddings = np.array(embeddings_list)
        else:
            embeddings = self.embedder.encode(words, show_progress_bar=False)

        return {word: emb for word, emb in zip(words, embeddings)}

    def _compute_pairwise_relatedness(
        self,
        words: Set[str],
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        ペアワイズ意味的関連性を計算（論文3.1節）

        Returns:
            {(word1, word2): relatedness_score}
        """
        relatedness = {}
        words_list = list(words)

        for i, w1 in enumerate(words_list):
            if w1 not in embeddings:
                continue
            for j, w2 in enumerate(words_list):
                if i >= j or w2 not in embeddings:  # 対称性を利用
                    continue

                # コサイン類似度
                sim = cosine_similarity(
                    embeddings[w1].reshape(1, -1),
                    embeddings[w2].reshape(1, -1)
                )[0, 0]

                # 両方向に保存
                relatedness[(w1, w2)] = sim
                relatedness[(w2, w1)] = sim

        return relatedness

    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        # 日本語の文分割
        sentences = re.split(r'[。！？\n]+', text)
        # 空文を除去
        return [s.strip() for s in sentences if s.strip()]

    def _build_semantic_graph(
        self,
        words: Set[str],
        relatedness: Dict[Tuple[str, str], float]
    ) -> nx.Graph:
        """
        意味的関連性グラフを構築（論文3.2.1節）

        Args:
            words: グラフに含める単語集合
            relatedness: ペアワイズ関連性スコア

        Returns:
            構築されたグラフ
        """
        graph = nx.Graph()

        # ノード追加
        for word in words:
            graph.add_node(word)

        # 各単語について強く関連する単語を選択
        for word in words:
            # 他の単語との関連性を取得
            related = []
            for other in words:
                if word != other and (word, other) in relatedness:
                    score = relatedness[(word, other)]
                    if score >= self.relmin:  # 最小閾値チェック
                        related.append((other, score))

            # 関連性でソートし上位reltop%を選択
            if related:
                related.sort(key=lambda x: x[1], reverse=True)
                top_k = max(1, int(len(related) * self.reltop))

                # エッジ追加
                for other, score in related[:top_k]:
                    graph.add_edge(word, other, weight=score)

        return graph

    def _personalized_pagerank(
        self,
        graph: nx.Graph,
        doc_words: Set[str],
        seed_terms: Set[str]
    ) -> Dict[str, float]:
        """
        Personalized PageRankを実行（論文3.2.2節）

        Args:
            graph: 意味的関連性グラフ
            doc_words: 文書に含まれる単語
            seed_terms: シード用語集合

        Returns:
            各ノードのPageRankスコア
        """
        if len(graph) == 0:
            return {}

        # シード用語から単語を抽出
        seed_words = self._extract_words_from_terms(seed_terms)

        # パーソナライゼーションベクトル初期化
        personalization = {}
        for node in graph.nodes():
            if node in seed_words:
                personalization[node] = 1.0
            else:
                personalization[node] = 0.0

        # 正規化（少なくとも1つのシード単語がある場合）
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        else:
            # シード単語がない場合は均等分布
            n = len(graph)
            personalization = {node: 1.0/n for node in graph.nodes()}

        # PageRank実行
        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=100,
                weight='weight'
            )
        except:
            # エラー時は均等分布を返す
            pagerank_scores = {node: 1.0/len(graph) for node in graph.nodes()}

        return pagerank_scores

    def _revise_scores(
        self,
        candidate_terms: Dict[str, float],
        word_importance: Dict[str, float],
        all_words: Set[str]
    ) -> Dict[str, float]:
        """
        候補用語のスコアを改訂（論文3.3節、式6）

        srk(ti) = (1.0 + Σ(nsmi(wx))/|words(ti)|) × nate(ti)

        Args:
            candidate_terms: 基底ATEスコア
            word_importance: 単語の意味的重要度
            all_words: すべての単語集合

        Returns:
            改訂されたスコア
        """
        # スコアの正規化
        ate_scores = list(candidate_terms.values())
        if not ate_scores:
            return {}

        max_ate = max(ate_scores)
        if max_ate == 0:
            max_ate = 1.0

        # 意味的重要度の正規化
        if word_importance:
            importance_values = list(word_importance.values())
            max_importance = max(importance_values) if importance_values else 1.0
        else:
            max_importance = 1.0

        revised_scores = {}

        for term, base_score in candidate_terms.items():
            # 基底スコアの正規化
            nate = base_score / max_ate

            # 用語を構成する単語を抽出
            term_words = self._extract_words_from_terms([term]) & all_words

            if term_words:
                # 構成単語の意味的重要度の平均
                sum_importance = 0.0
                for word in term_words:
                    if word in word_importance and max_importance > 0:
                        nsmi = word_importance[word] / max_importance
                        sum_importance += nsmi

                avg_importance = sum_importance / len(term_words)
            else:
                avg_importance = 0.0

            # 式6によるスコア改訂
            revised_scores[term] = (1.0 + avg_importance) * nate

        return revised_scores


class SemReRankExtractor(BaseExtractor):
    """
    SemRe-Rankを使用した専門用語抽出器
    """

    def __init__(
        self,
        base_ate_method: str = "tfidf",  # 基底ATE手法
        min_frequency: int = 2,
        min_term_length: int = 2,
        max_term_length: int = 10,
        **semrerank_params
    ):
        super().__init__()
        self.base_ate_method = base_ate_method
        self.min_frequency = min_frequency
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.semrerank = SemReRank(**semrerank_params)

    def extract(self, text: str, **kwargs) -> List[Term]:
        """
        SemRe-Rankによる専門用語抽出
        """
        # 1. 候補用語を抽出
        candidates = self._extract_candidates(text)

        if not candidates:
            return []

        # 2. Stage A: シード選定用スコア（C-value重視: 0.3 TF-IDF + 0.7 C-value）
        seed_scores = self._calculate_base_scores(text, candidates, stage="seed")
        
        # 3. Stage B: 最終ランク付け用スコア（TF-IDF重視: 0.7 TF-IDF + 0.3 C-value）
        base_scores = self._calculate_base_scores(text, candidates, stage="final")

        # 4. SemRe-Rankでスコアを改善（seed_scoresをシード選定に使用）
        enhanced_scores = self.semrerank.enhance_ate_scores(
            text,
            base_scores,
            candidates,
            seed_selection_scores=seed_scores  # シード選定用の別スコアを渡す
        )

        # 5. Termオブジェクトに変換
        terms = []
        for term, score in enhanced_scores.items():
            terms.append(Term(
                term=term,
                score=score,
                definition="",
                metadata={
                    "frequency": candidates.get(term, 0),
                    "seed_score": seed_scores.get(term, 0.0),
                    "base_score": base_scores.get(term, 0.0),
                    "enhanced_score": score,
                    "method": "SemRe-Rank"
                }
            ))

        # 6. スコアでソートして返す
        terms.sort(key=lambda x: x.score, reverse=True)
        return terms[:100]  # 上位100件を返す  # 上位100件を返す

    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """候補用語を抽出し頻度をカウント"""
        candidates = defaultdict(int)

        # 複合語パターン（日本語対応）
        patterns = [
            r'[ァ-ヶー]+',  # カタカナ
            r'[一-龯]{2,}',  # 漢字（2文字以上）
            r'[A-Z][A-Za-z0-9]*',  # 英単語
            r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',  # カタカナ+漢字
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group()
                if self.min_term_length <= len(term) <= self.max_term_length:
                    candidates[term] += 1

        # 最小頻度でフィルタ
        filtered = {
            term: freq
            for term, freq in candidates.items()
            if freq >= self.min_frequency
        }

        return filtered

    def _calculate_base_scores(
        self,
        text: str,
        candidates: Dict[str, int],
        stage: str = "final"  # "seed" or "final"
    ) -> Dict[str, float]:
        """基底ATEスコアを計算（TF-IDFとC-valueの重み付き組み合わせ）

        Args:
            text: 対象テキスト
            candidates: 候補用語と頻度
            stage: "seed" (Stage A: 0.3 TF-IDF + 0.7 C-value) or
                   "final" (Stage B: 0.7 TF-IDF + 0.3 C-value)
        """
        # TF-IDFとC-valueを両方計算
        tfidf_scores = self._calculate_tfidf(text, candidates)
        cvalue_scores = self._calculate_cvalue(candidates)

        # 通常のmin-max正規化
        tfidf_normalized = self._min_max_normalize(tfidf_scores)
        cvalue_normalized = self._min_max_normalize(cvalue_scores)

        # 段階別の重み設定
        if stage == "seed":
            # Stage A: シード選定 - C-value重視
            tfidf_weight = 0.3
            cvalue_weight = 0.7
        else:
            # Stage B: 最終ランク付け - TF-IDF重視
            tfidf_weight = 0.7
            cvalue_weight = 0.3

        # 重み付き結合
        combined_scores = {}
        for term in candidates:
            tfidf = tfidf_normalized.get(term, 0.0)
            cvalue = cvalue_normalized.get(term, 0.0)
            combined_scores[term] = tfidf_weight * tfidf + cvalue_weight * cvalue

        # 再正規化は行わない（0スコアを避けるため）
        return combined_scores

    def _min_max_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """通常のmin-max正規化

        Args:
            scores: 正規化前のスコア辞書

        Returns:
            正規化後のスコア辞書（0-1の範囲）
        """
        if not scores:
            return {}

        values = list(scores.values())
        if len(values) == 1:
            return {k: 1.0 for k in scores}

        min_val = min(values)
        max_val = max(values)

        # 範囲が0の場合の処理
        if max_val - min_val < 1e-10:
            return {k: 0.5 for k in scores}

        # 正規化
        normalized = {}
        for term, score in scores.items():
            normalized[term] = (score - min_val) / (max_val - min_val)

        return normalized

    def _calculate_tfidf(
        self,
        text: str,
        candidates: Dict[str, int]
    ) -> Dict[str, float]:
        """TF-IDFスコアを計算"""
        # 文書を文に分割
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            sentences = [text]

        # 文書頻度を計算
        doc_frequencies = defaultdict(int)
        for term in candidates:
            for sent in sentences:
                if term in sent:
                    doc_frequencies[term] += 1

        # TF-IDF計算
        tfidf_scores = {}
        total_docs = len(sentences)
        doc_length = len(text)

        for term, freq in candidates.items():
            tf = freq / doc_length
            idf = math.log((total_docs + 1) / (doc_frequencies[term] + 1)) + 1
            tfidf_scores[term] = tf * idf

        return tfidf_scores

    def _calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """C-valueスコアを計算"""
        c_values = {}

        for candidate, freq in candidates.items():
            length = len(candidate)

            # より長い候補語を探す
            longer_terms = []
            for other in candidates:
                if other != candidate and candidate in other:
                    longer_terms.append(other)

            # C-value計算
            if not longer_terms:
                c_value = math.log2(max(length, 2)) * freq
            else:
                sum_freq = sum(candidates[term] for term in longer_terms)
                t_a = len(longer_terms)
                c_value = math.log2(max(length, 2)) * (freq - sum_freq / t_a)

            c_values[candidate] = max(c_value, 0.0)

        return c_values


# 使用例
if __name__ == "__main__":
    # デモテキスト
    text = """
    アンモニア燃料エンジンは、次世代の環境対応技術として注目されている。
    このアンモニア燃料エンジンは、従来のディーゼルエンジンと比較して、
    CO2排出量を大幅に削減できる。アンモニア燃料レシプロエンジンの開発も進んでおり、
    舶用アンモニアエンジンとしての実用化が期待されている。
    """

    # SemRe-Rank抽出器を初期化（自動シード選択モード）
    extractor = SemReRankExtractor(
        base_ate_method="tfidf",
        auto_select_seeds=True,  # デモのため自動選択
        seed_z=50
    )

    # 用語抽出
    terms = extractor.extract(text)

    # 結果表示
    print("\n抽出された専門用語（SemRe-Rank）:")
    print("-" * 50)
    for i, term in enumerate(terms[:20], 1):
        print(f"{i:2}. {term.term:20} (Score: {term.score:.4f})")