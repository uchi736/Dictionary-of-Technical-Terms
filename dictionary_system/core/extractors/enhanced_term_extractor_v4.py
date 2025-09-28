#!/usr/bin/env python3
"""
専門用語抽出システム V4
====================================
V3の改良版: 正しいSemRe-Rank実装 + RAG + 階層的類義語抽出

改善点:
- V3の誤ったkNNグラフ → SemRe-Rank論文準拠の意味的関連性グラフ
- TF-IDF/C-valueスコアからシード自動選定
- 定義ベースLLMフィルタリング
- 階層的類義語抽出 (HDBSCAN + カテゴリ名生成)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import logging
import re
import os
import math
import hashlib
import pickle
from dotenv import load_dotenv

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from dictionary_system.core.models.base_extractor import BaseExtractor, Term

# Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

# RAG拡張機能
from dictionary_system.core.rag import (
    enrich_terms_with_definitions,
    filter_technical_terms_by_definition,
    extract_synonym_hierarchy
)

# SudachiPy
try:
    from sudachipy import tokenizer, dictionary
    SUDACHI_AVAILABLE = True
except ImportError:
    SUDACHI_AVAILABLE = False

# PyMuPDF
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


class EnhancedTermExtractorV4(BaseExtractor):
    """
    統合版専門用語抽出器 V4

    特徴:
    - V3のTF-IDF/C-value計算を継承
    - 正しいSemRe-Rankアルゴリズム
    - PostgreSQL + RAG統合
    - 定義ベースフィルタリング
    - 階層的類義語抽出
    """

    def __init__(
        self,
        # 基本パラメータ
        min_term_length: int = 2,
        max_term_length: int = 15,
        min_frequency: int = 2,
        # 候補語抽出パラメータ
        use_sudachi: bool = True,
        use_ngram_generation: bool = True,
        max_ngram: int = 3,
        # SemRe-Rankパラメータ
        relmin: float = 0.5,
        reltop: float = 0.15,
        alpha: float = 0.85,
        seed_z: int = 50,
        auto_select_seeds: bool = True,
        use_elbow_detection: bool = True,
        min_seed_count: int = 5,
        max_seed_ratio: float = 0.7,
        # 埋め込み
        use_cache: bool = True,
        cache_dir: str = "cache/embeddings",
        # RAG/Azure OpenAI
        use_rag_context: bool = True,
        use_azure_openai: bool = True,
        db_url: Optional[str] = None,
        collection_name: str = "documents",
        # 拡張機能
        enable_definition_generation: bool = True,
        enable_definition_filtering: bool = True,
        enable_synonym_hierarchy: bool = True,
        top_n_definition: int = 30,
        min_cluster_size: int = 2,
        generate_category_names: bool = True
    ):
        """
        初期化
        """
        super().__init__()

        # 基本設定
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.min_frequency = min_frequency

        # 候補語抽出設定
        self.use_sudachi = use_sudachi
        self.use_ngram_generation = use_ngram_generation
        self.max_ngram = max_ngram

        # SemRe-Rank設定
        self.relmin = relmin
        self.reltop = reltop
        self.alpha = alpha
        self.seed_z = seed_z
        self.auto_select_seeds = auto_select_seeds
        self.use_elbow_detection = use_elbow_detection
        self.min_seed_count = min_seed_count
        self.max_seed_ratio = max_seed_ratio

        # キャッシュ設定
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # RAG/Azure OpenAI設定
        self.use_rag_context = use_rag_context
        self.use_azure_openai = use_azure_openai
        self.collection_name = collection_name

        # 拡張機能設定
        self.enable_definition_generation = enable_definition_generation
        self.enable_definition_filtering = enable_definition_filtering
        self.enable_synonym_hierarchy = enable_synonym_hierarchy
        self.top_n_definition = top_n_definition
        self.min_cluster_size = min_cluster_size
        self.generate_category_names = generate_category_names

        # Azure OpenAI初期化
        if use_azure_openai:
            self._setup_azure_openai()

        # RAG初期化
        if use_rag_context and db_url:
            self._setup_rag_store(db_url)

        # SudachiPy初期化
        if SUDACHI_AVAILABLE:
            self.sudachi_tokenizer = dictionary.Dictionary().create()
            self.sudachi_mode = tokenizer.Tokenizer.SplitMode.C
            logger.info("SudachiPy initialized")

    def _setup_azure_openai(self):
        """Azure OpenAI設定"""
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not all([self.azure_endpoint, self.azure_api_key]):
            logger.warning("Azure OpenAI not configured")
            self.use_azure_openai = False
            return

        # Embeddings
        self.azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_deployment="text-embedding-3-small"
        )

        logger.info("Azure OpenAI initialized")

    def _setup_rag_store(self, db_url: str):
        """RAGベクトルストア設定"""
        try:
            self.vector_store = PGVector(
                collection_name=self.collection_name,
                connection_string=db_url,
                embedding_function=self.azure_embeddings,
                pre_delete_collection=False
            )
            self.db_engine = create_engine(db_url)
            logger.info(f"RAG store connected: {self.collection_name}")
        except Exception as e:
            logger.warning(f"RAG store connection failed: {e}")
            self.use_rag_context = False


    def extract(self, text: str) -> List[Term]:
        """
        専門用語抽出メインフロー

        Args:
            text: 入力テキスト

        Returns:
            抽出された専門用語リスト
        """
        logger.info("=== V4 Term Extraction Started ===")

        # STEP 1: 候補語抽出 (V3継承)
        logger.info("STEP 1: Candidate extraction")
        candidates = self._extract_candidates(text)
        logger.info(f"Extracted {len(candidates)} candidates")

        if not candidates:
            return []

        # STEP 2: TF-IDF計算 (V3継承)
        logger.info("STEP 2: TF-IDF calculation")
        terms_list = list(candidates.keys())
        tfidf_scores = self._calculate_tfidf(text, terms_list)

        # STEP 3: C-value計算 (V3継承)
        logger.info("STEP 3: C-value calculation")
        cvalue_scores = self._calculate_cvalue(candidates)

        # 基底スコア統合 (TF-IDF + C-value)
        base_scores = {}
        for term in terms_list:
            tfidf = tfidf_scores.get(term, 0.0)
            cval = cvalue_scores.get(term, 0.0)
            base_scores[term] = 0.7 * tfidf + 0.3 * cval

        # STEP 4-6: SemRe-Rankアルゴリズム
        logger.info("STEP 4-6: SemRe-Rank algorithm")
        final_scores = self._semrerank_scoring(
            terms_list,
            base_scores
        )

        # Term オブジェクトに変換
        terms = [
            Term(term=term, score=score)
            for term, score in final_scores.items()
        ]

        # スコア順にソート
        terms.sort(key=lambda t: t.score, reverse=True)

        logger.info(f"Extracted {len(terms)} terms")

        # STEP 7: RAG定義生成 (オプション)
        if self.enable_definition_generation:
            logger.info("STEP 7: RAG definition generation")
            terms = enrich_terms_with_definitions(
                terms=terms,
                text=text,
                top_n=self.top_n_definition,
                verbose=False
            )

        # STEP 8: LLM専門用語判定 (オプション)
        if self.enable_definition_filtering:
            logger.info("STEP 8: LLM technical term filtering")
            terms = filter_technical_terms_by_definition(
                terms,
                verbose=False
            )

        # STEP 9: 階層的類義語抽出 (オプション)
        if self.enable_synonym_hierarchy and len(terms) >= self.min_cluster_size:
            logger.info("STEP 9: Hierarchical synonym extraction")
            self.hierarchy = extract_synonym_hierarchy(
                terms,
                min_cluster_size=self.min_cluster_size,
                generate_category_names=self.generate_category_names,
                verbose=False
            )
            # 階層情報をメタデータに追加
            for term in terms:
                for rep, node in self.hierarchy.items():
                    if term.term in node.terms:
                        term.metadata['cluster_id'] = node.cluster_id
                        term.metadata['cluster_category'] = node.category_name
                        break
        else:
            self.hierarchy = None

        return terms

    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """
        候補語抽出 (正規表現 + SudachiPy + n-gram)

        Returns:
            {用語: 出現頻度}
        """
        candidates = Counter()

        # 1. 正規表現パターンで抽出
        patterns = [
            r'[ァ-ヶー]+',  # カタカナ
            r'[一-龯]{2,}',  # 漢字（2文字以上）
            r'[0-9]*[A-Za-z][A-Za-z0-9]*',  # 英数字（数字始まり可、最低1文字必須）
            r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',  # カタカナ+漢字
            r'[一-龯]+[ァ-ヶー]+[一-龯]+',  # 漢字+カタカナ+漢字
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group()

                # 純粋な数字のみは除外
                if re.match(r'^\d+$', term):
                    continue

                # 長さチェック
                if self.min_term_length <= len(term) <= self.max_term_length:
                    candidates[term] += 1

        # 2. SudachiPy形態素解析による抽出
        if self.use_sudachi and SUDACHI_AVAILABLE:
            sudachi_candidates = self._extract_candidates_with_sudachi(text)
            candidates.update(sudachi_candidates)

        # 3. n-gram複合語生成
        if self.use_ngram_generation and SUDACHI_AVAILABLE:
            ngram_candidates = self._generate_ngram_candidates(text)
            candidates.update(ngram_candidates)

        # 4. 見出しから抽出
        heading_terms = self._extract_headings(text)
        for term in heading_terms:
            if term not in candidates:
                candidates[term] = 1
            candidates[term] += 2  # ボーナス

        # 5. 頻度フィルタ
        filtered = {}
        for term, freq in candidates.items():
            if freq >= self.min_frequency:
                filtered[term] = freq
            elif freq == 1 and term in heading_terms:
                # 見出しに含まれる用語は頻度1でも残す
                filtered[term] = freq

        return filtered

    def _extract_headings(self, text: str) -> Set[str]:
        """見出しから用語抽出"""
        heading_terms = set()
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',
            r'^\d+\.\s+(.+)$',
            r'^第[一二三四五六七八九十\d]+[章節項]\s*(.+)$',
        ]

        for line in text.split('\n'):
            line = line.strip()
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    heading_text = match.group(1)
                    # カタカナ・漢字抽出
                    terms = re.findall(r'[ァ-ヶー一-龯]{2,}', heading_text)
                    for term in terms:
                        if self.min_term_length <= len(term) <= self.max_term_length:
                            heading_terms.add(term)
                    break

        return heading_terms

    def _extract_candidates_with_sudachi(self, text: str) -> Dict[str, int]:
        """
        SudachiPyによる形態素解析ベースの候補語抽出

        Returns:
            {用語: 出現頻度}
        """
        if not SUDACHI_AVAILABLE or not hasattr(self, 'sudachi_tokenizer'):
            return {}

        candidates = Counter()

        # 形態素解析
        tokens = self.sudachi_tokenizer.tokenize(text, self.sudachi_mode)

        # 名詞句抽出
        current_phrase = []
        for token in tokens:
            pos = token.part_of_speech()[0]

            if pos in ['名詞', '接頭辞']:
                current_phrase.append(token.surface())
            else:
                if current_phrase:
                    # 名詞句全体を追加
                    phrase = ''.join(current_phrase)
                    if self.min_term_length <= len(phrase) <= self.max_term_length:
                        candidates[phrase] += 1

                    # 後方suffix生成（部分的な組み合わせ）
                    if len(current_phrase) > 2:
                        for i in range(1, len(current_phrase)):
                            sub_phrase = ''.join(current_phrase[i:])
                            if self.min_term_length <= len(sub_phrase) <= self.max_term_length:
                                candidates[sub_phrase] += 1

                    current_phrase = []

        # 最後の句を処理
        if current_phrase:
            phrase = ''.join(current_phrase)
            if self.min_term_length <= len(phrase) <= self.max_term_length:
                candidates[phrase] += 1

        return candidates

    def _generate_ngram_candidates(self, text: str) -> Dict[str, int]:
        """
        形態素n-gramによる複合語生成

        Args:
            text: 入力テキスト

        Returns:
            {用語: 出現頻度}
        """
        if not SUDACHI_AVAILABLE or not hasattr(self, 'sudachi_tokenizer'):
            return {}

        candidates = Counter()

        # 形態素解析
        tokens = self.sudachi_tokenizer.tokenize(text, self.sudachi_mode)

        # 名詞のみ抽出
        nouns = [t.surface() for t in tokens if t.part_of_speech()[0] == '名詞']

        if len(nouns) < 2:
            return candidates

        # n-gram生成（2-gram から max_ngram まで）
        for window_size in range(2, min(self.max_ngram + 1, len(nouns) + 1)):
            for i in range(len(nouns) - window_size + 1):
                phrase = ''.join(nouns[i:i + window_size])
                if self.min_term_length <= len(phrase) <= self.max_term_length:
                    candidates[phrase] += 1

        return candidates

    def _calculate_tfidf(self, text: str, terms: List[str]) -> Dict[str, float]:
        """
        TF-IDF計算 (V3から継承)

        Returns:
            {用語: TF-IDFスコア}
        """
        if not terms or not text:
            return {term: 0.0 for term in terms}

        doc_length = len(text)
        term_frequencies = {}

        for term in terms:
            count = text.count(term)
            term_frequencies[term] = count

        # 文単位で分割
        sentences = text.replace('\n', '。').split('。')
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            sentences = [text]

        doc_frequencies = {}
        for term in terms:
            doc_count = sum(1 for sent in sentences if term in sent)
            doc_frequencies[term] = max(doc_count, 1)

        # TF-IDF計算
        tfidf_scores = {}
        total_docs = len(sentences)

        for term in terms:
            tf = (term_frequencies[term] + 0.5) / (doc_length + 1.0)
            idf = max(0.1, math.log((total_docs + 1) / (doc_frequencies[term] + 1)) + 1)
            tfidf_scores[term] = tf * idf

        return tfidf_scores

    def _calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """
        C-value計算 (V3から継承)

        Returns:
            {用語: C-valueスコア}
        """
        cvalue_scores = {}
        term_list = list(candidates.keys())

        for term in term_list:
            freq = candidates[term]
            length = len(term)

            # ネストカウント
            nested_count = 0
            for other_term in term_list:
                if term != other_term and term in other_term:
                    nested_count += 1

            # C-value計算
            if nested_count == 0:
                cvalue = math.log(length + 1) * freq
            else:
                avg_freq = sum(candidates[t] for t in term_list if term in t and term != t) / nested_count
                cvalue = math.log(length + 1) * (freq - (1.0 / nested_count) * avg_freq)

            cvalue_scores[term] = max(cvalue, 0.0)

        return cvalue_scores

    def _semrerank_scoring(
        self,
        terms: List[str],
        base_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        SemRe-Rankスコア計算

        Args:
            terms: 用語リスト
            base_scores: 基底スコア (TF-IDF + C-value)

        Returns:
            {用語: 最終スコア}
        """
        if not terms:
            return {}

        # 埋め込み計算
        embeddings = self._compute_embeddings(terms)

        # シード選定
        seeds = self._select_seeds(terms, base_scores)
        logger.info(f"Selected {len(seeds)} seed terms")

        # 意味的関連性グラフ構築
        graph = self._build_semantic_graph(terms, embeddings, seeds)
        logger.info(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

        # Personalized PageRank
        ppr_scores = self._personalized_pagerank(graph, seeds)

        # 最終スコア = base_score × ppr_score
        final_scores = {}
        for term in terms:
            base = base_scores.get(term, 0.01)
            ppr = ppr_scores.get(term, 0.5)
            final_scores[term] = base * ppr

        return final_scores

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """埋め込み計算 (キャッシュ対応)"""
        if self.use_cache:
            cache_key = hashlib.md5(str(texts).encode()).hexdigest()
            cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Azure Embeddings
        if self.use_azure_openai and hasattr(self, 'azure_embeddings'):
            embeddings = self.azure_embeddings.embed_documents(texts)
            embeddings = np.array(embeddings)
        else:
            # フォールバック (SentenceTransformer)
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embeddings = embedder.encode(texts, show_progress_bar=False)

        # キャッシュ保存
        if self.use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)

        return embeddings

    def _select_seeds(
        self,
        terms: List[str],
        base_scores: Dict[str, float]
    ) -> List[str]:
        """シード選定 (TF-IDF/C-valueスコアからエルボー法)"""
        if not self.auto_select_seeds:
            return []

        # スコア順にソート
        sorted_terms = sorted(terms, key=lambda t: base_scores.get(t, 0), reverse=True)
        top_z = sorted_terms[:self.seed_z]

        if not self.use_elbow_detection:
            # エルボー法なし: 上位固定数
            return top_z[:self.min_seed_count]

        # エルボー法でシード数決定
        scores = [base_scores.get(t, 0) for t in top_z]
        diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]

        max_seeds = int(len(top_z) * self.max_seed_ratio)
        elbow_idx = self.min_seed_count

        for i in range(self.min_seed_count, min(len(diffs), max_seeds)):
            if i > 0 and diffs[i] < diffs[i-1] * 0.5:
                elbow_idx = i
                break

        return top_z[:elbow_idx]

    def _build_semantic_graph(
        self,
        terms: List[str],
        embeddings: np.ndarray,
        seeds: List[str]
    ) -> nx.Graph:
        """意味的関連性グラフ構築 (SemRe-Rank論文準拠)"""
        graph = nx.Graph()

        # 類似度行列計算
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        n = len(terms)
        for i in range(n):
            graph.add_node(terms[i])

        # relmin閾値フィルタ + reltop上位選択
        for i in range(n):
            sims = [(j, sim_matrix[i][j]) for j in range(n) if i != j]
            sims.sort(key=lambda x: x[1], reverse=True)

            # relmin閾値
            sims = [(j, s) for j, s in sims if s >= self.relmin]

            # reltop上位
            top_k = max(1, int(len(sims) * self.reltop))
            sims = sims[:top_k]

            # エッジ追加
            for j, sim in sims:
                graph.add_edge(terms[i], terms[j], weight=float(sim))

        return graph

    def _personalized_pagerank(
        self,
        graph: nx.Graph,
        seeds: List[str]
    ) -> Dict[str, float]:
        """Personalized PageRank (シードベース)"""
        if len(graph) == 0:
            return {}

        # シード用語を初期化
        personalization = {}
        for node in graph.nodes():
            if node in seeds:
                personalization[node] = 1.0
            else:
                personalization[node] = 0.0

        # 正規化
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        else:
            personalization = {node: 1.0/len(graph) for node in graph.nodes()}

        # PageRank実行
        try:
            ppr_scores = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=100,
                weight='weight'
            )
        except:
            ppr_scores = {node: 1.0 / len(graph) for node in graph.nodes()}

        return ppr_scores