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
        seed_z: int = 10,
        auto_select_seeds: bool = True,
        use_elbow_detection: bool = False,
        min_seed_count: int = 5,
        max_seed_ratio: float = 0.2,
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
        generate_category_names: bool = True,
        # UMAP次元削減
        use_umap: bool = False,
        umap_n_components: int = 50
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

        # UMAP設定
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components

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

        # STEP 1.5: 見出し用語抽出（新規）
        logger.info("STEP 1.5: Header term extraction")
        header_terms = self._extract_header_terms(text)

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

        # STEP 3.3: 見出しボーナス適用（新規）
        logger.info("STEP 3.3: Applying header bonus")
        base_scores = self._apply_header_bonus(base_scores, header_terms)

        # STEP 3.5: 複合度ボーナス適用
        logger.info("STEP 3.5: Applying complexity bonus")
        base_scores = self._apply_complexity_bonus(base_scores)

        # STEP 4-6: SemRe-Rankアルゴリズム
        logger.info("STEP 4-6: SemRe-Rank algorithm")
        terms_list = list(base_scores.keys())
        final_scores = self._semrerank_scoring(
            terms_list,
            base_scores
        )

        # STEP 6.5: 軽い部分文字列フィルタリング (10%以下の明らかな重複のみ除外)
        logger.info("STEP 6.5: Light substring filtering (10% threshold)")
        final_scores = self._filter_substring_duplicates(final_scores, score_ratio_threshold=0.1)

        # Term オブジェクトに変換
        terms = [
            Term(term=term, score=score)
            for term, score in final_scores.items()
        ]

        # スコア順にソート
        terms.sort(key=lambda t: t.score, reverse=True)

        logger.info(f"After filtering: {len(terms)} terms")

        # 定義生成数を動的に計算
        definition_count = self._calculate_definition_count(len(terms))

        # STEP 7: RAG定義生成 (上位N件)
        if self.enable_definition_generation:
            logger.info(f"STEP 7: RAG definition generation (top {definition_count} terms)")
            terms_for_definition = terms[:definition_count]
            terms_with_definition = enrich_terms_with_definitions(
                terms=terms_for_definition,
                text=text,
                verbose=False
            )
            # 定義生成されなかった残りの用語と結合
            terms = terms_with_definition + terms[definition_count:]

        # STEP 8: LLM専門用語判定（定義がある用語のみ）
        if self.enable_definition_filtering:
            logger.info("STEP 8: LLM technical term filtering")
            # 定義がある用語のみフィルタリング
            terms_with_def = [t for t in terms if t.definition]
            technical_terms = filter_technical_terms_by_definition(
                terms_with_def,
                verbose=False
            )
        else:
            technical_terms = terms[:definition_count]  # 定義生成した分のみ

        # STEP 9: 階層的類義語抽出（専門用語確定後）
        if self.enable_synonym_hierarchy and len(technical_terms) >= self.min_cluster_size:
            logger.info(f"STEP 9: Hierarchical clustering ({len(technical_terms)} technical terms)")
            self.hierarchy = extract_synonym_hierarchy(
                technical_terms,
                min_cluster_size=self.min_cluster_size,
                generate_category_names=False,
                use_umap=self.use_umap,
                umap_n_components=self.umap_n_components,
                verbose=False
            )
        else:
            self.hierarchy = None
            logger.info(f"Skipping clustering: only {len(technical_terms)} terms (min={self.min_cluster_size})")

        # STEP 10: カテゴリ名生成（クラスタリング後）
        if self.hierarchy and self.generate_category_names:
            logger.info("STEP 10: Category name generation")
            from dictionary_system.core.rag.synonym_extractor import generate_cluster_category_names
            self.hierarchy = generate_cluster_category_names(
                self.hierarchy,
                technical_terms,
                verbose=False
            )

        # 階層情報をメタデータに追加
        if self.hierarchy:
            for term in technical_terms:
                for rep, node in self.hierarchy.items():
                    if term.term in node.terms:
                        term.metadata['cluster_id'] = node.cluster_id
                        term.metadata['cluster_category'] = node.category_name
                        break

        return technical_terms

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
            r'[0-9]+[A-Za-z]+[0-9A-Za-z~\-]*',  # 型式番号（6DE~~, 6DE-50など）
            r'[A-Za-z]+[0-9]+[A-Za-z0-9~\-]*',  # 逆パターン（ABC123など）
            r'[A-Za-z]+',  # 純粋な英字（2文字以上はmin_term_lengthで制御）
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
        heading_terms = self._extract_header_terms(text)
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

    def _extract_header_terms(self, text: str) -> Set[str]:
        """
        見出しから重要用語を抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            見出しに含まれる用語のセット
        """
        header_patterns = [
            # 日本語見出しパターン
            r'^第[０-９\d]+[章節項]\s+(.+)$',     # 第1章 エンジンの構造
            r'^[０-９\d]+\.\s+(.+)$',            # 1. はじめに
            r'^[０-９\d]+\.[０-９\d]+\s+(.+)$',   # 1.1 背景
            r'^\[(.+?)\]',                       # [重要事項]
            r'^【(.+?)】',                        # 【注意】
            # 英語見出し
            r'^Chapter\s+\d+[:：]\s*(.+)$',      # Chapter 1: Introduction
            r'^Section\s+\d+[:：]\s*(.+)$',      # Section 2: Methods
            # 強調パターン（大文字や記号）
            r'^[■□▼▲●○]\s*(.+)$',             # ■ 概要
            r'^[A-Z\s]{5,}$',                    # IMPORTANT NOTICE
        ]
        
        header_terms = set()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in header_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    header_text = match.group(1) if match.groups() else line
                    
                    # 見出しテキストから用語候補を抽出
                    # 既存のパターンを使用
                    patterns = [
                        r'[ァ-ヶー]+',  # カタカナ
                        r'[一-龯]{2,}',  # 漢字（2文字以上）
                        r'[0-9]+[A-Za-z]+[0-9A-Za-z~\-]*',  # 型式番号
                        r'[A-Za-z]+[0-9]+[A-Za-z0-9~\-]*',  # 逆パターン
                        r'[A-Za-z]{2,}',  # 英字（2文字以上）
                    ]
                    
                    for p in patterns:
                        terms = re.findall(p, header_text)
                        header_terms.update(terms)
                    break
        
        logger.info(f"Extracted {len(header_terms)} terms from headers")
        return header_terms
    
    def _apply_header_bonus(
        self,
        base_scores: Dict[str, float],
        header_terms: Set[str]
    ) -> Dict[str, float]:
        """
        見出しに含まれる用語に2倍ボーナスを適用
        
        Args:
            base_scores: 基底スコア辞書
            header_terms: 見出し用語のセット
            
        Returns:
            ボーナス適用後のスコア辞書
        """
        adjusted_scores = base_scores.copy()
        bonus_applied = 0
        
        for term in adjusted_scores:
            if term in header_terms:
                adjusted_scores[term] *= 2.0
                bonus_applied += 1
        
        if bonus_applied > 0:
            logger.info(f"Header bonus applied to {bonus_applied} terms")
            
        return adjusted_scores

    def _calculate_definition_count(self, candidate_count: int) -> int:
        """
        文書規模に応じた定義生成数を計算
        
        Args:
            candidate_count: 候補用語数
            
        Returns:
            定義生成数
        """
        percentage = 0.25  # 上位25%
        min_count = 15     # 最小15件
        max_count = 50     # 最大50件
        
        target = int(candidate_count * percentage)
        result = max(min_count, min(target, max_count))
        
        logger.info(f"Definition count: {result} (from {candidate_count} candidates)")
        return result


    def _extract_candidates_with_sudachi(self, text: str) -> Dict[str, int]:
        """
        SudachiPyによる形態素解析ベースの候補語抽出
        連続名詞を連結するのみ（部分列生成はn-gramに任せる）

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
                    # 名詞句全体のみを追加（後方suffixは削除）
                    phrase = ''.join(current_phrase)
                    if self.min_term_length <= len(phrase) <= self.max_term_length:
                        candidates[phrase] += 1

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

    def _filter_substring_duplicates(
        self,
        scored_terms: Dict[str, float],
        score_ratio_threshold: float = 0.2
    ) -> Dict[str, float]:
        """
        部分文字列の重複を除去

        スコアが大幅に低い部分文字列を除外
        例: 「削減」(0.048) vs 「削減率」(0.714)
            → 「削減」はスコアが低いため除外

        Args:
            scored_terms: {用語: スコア}
            score_ratio_threshold: より長い用語のスコアに対する閾値（デフォルト0.2 = 20%）

        Returns:
            フィルタ後の{用語: スコア}
        """
        terms_sorted = sorted(scored_terms.items(), key=lambda x: len(x[0]), reverse=True)
        to_remove = set()

        for i, (term_long, score_long) in enumerate(terms_sorted):
            if term_long in to_remove:
                continue

            for j in range(i + 1, len(terms_sorted)):
                term_short, score_short = terms_sorted[j]

                if term_short in to_remove:
                    continue

                if term_short in term_long:
                    if score_short < score_long * score_ratio_threshold:
                        to_remove.add(term_short)
                        logger.debug(
                            f"Removed substring: '{term_short}' (score={score_short:.3f}) "
                            f"contained in '{term_long}' (score={score_long:.3f})"
                        )

        filtered = {term: score for term, score in scored_terms.items() if term not in to_remove}
        logger.info(f"Substring filtering: {len(scored_terms)} → {len(filtered)} terms")

        return filtered

    def _apply_complexity_bonus(self, base_scores: Dict[str, float]) -> Dict[str, float]:
        """
        複合語に対してボーナススコアを付与
        
        Args:
            base_scores: 基底スコア辞書
            
        Returns:
            複合度ボーナス適用後のスコア辞書
        """
        adjusted_scores = {}
        for term, score in base_scores.items():
            # カタカナ部、漢字部、英数部などの構成要素数を計算
            components = len(re.findall(
                r'[ァ-ヶー]+|[一-龯]+|[A-Za-z0-9]+|[0-9]+',
                term
            ))
            # 複合度に応じてボーナス（単体なら1.0、複合なら1.3, 1.6...）
            complexity_multiplier = 1 + 0.3 * max(0, components - 1)
            adjusted_scores[term] = score * complexity_multiplier
            
        return adjusted_scores

    def _early_filtering(
        self,
        base_scores: Dict[str, float],
        percentile_threshold: int = 20
    ) -> Dict[str, float]:
        """
        早期フィルタリング（SemRe-Rank前に明らかな低スコアを除外）
        
        Args:
            base_scores: 基底スコア辞書
            percentile_threshold: 下位何%を除外するか（デフォルト20%）
            
        Returns:
            フィルタリング後のスコア辞書
        """
        if not base_scores:
            return base_scores
            
        import numpy as np
        score_values = list(base_scores.values())
        threshold = np.percentile(score_values, percentile_threshold)
        
        # 閾値以上のスコアを持つ用語のみ残す
        filtered_scores = {
            term: score
            for term, score in base_scores.items()
            if score >= threshold
        }
        
        logger.info(f"Early filtering: {len(base_scores)} → {len(filtered_scores)} terms")
        return filtered_scores

    def _filter_terms_within_clusters(
        self,
        terms: List[Term],
        hierarchy: Dict,
        score_ratio_threshold: float = 3.0,
        llm_model: str = "gpt-4.1-mini"
    ) -> List[Term]:
        """
        クラスタ内で類似用語をフィルタリング

        各クラスタについて:
        - 単独用語: そのまま残す
        - スコア差が大きい（threshold倍以上）: 上位のみ残す
        - スコア拮抗: LLMに判定依頼

        Args:
            terms: Term オブジェクトのリスト
            hierarchy: クラスタ階層情報
            score_ratio_threshold: スコア差の閾値（デフォルト3.0 = 3倍）
            llm_model: LLM モデル名

        Returns:
            フィルタ後のTermリスト
        """
        from dictionary_system.config.rag_config import Config
        from langchain_openai import AzureChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from dictionary_system.config.prompts import get_cluster_term_filtering_prompt_messages
        import json

        if not hierarchy:
            return terms

        config = Config()
        terms_to_keep = []
        llm_call_count = 0

        for rep, node in hierarchy.items():
            cluster_terms = [t for t in terms if t.term in node.terms]

            if not cluster_terms:
                continue

            if len(cluster_terms) == 1:
                terms_to_keep.extend(cluster_terms)
                logger.debug(f"Cluster '{node.category_name}': Single term, keeping '{cluster_terms[0].term}'")
                continue

            cluster_terms_sorted = sorted(cluster_terms, key=lambda t: t.score, reverse=True)
            top_score = cluster_terms_sorted[0].score
            second_score = cluster_terms_sorted[1].score if len(cluster_terms_sorted) > 1 else 0

            if top_score > second_score * score_ratio_threshold:
                terms_to_keep.append(cluster_terms_sorted[0])
                logger.info(
                    f"Cluster '{node.category_name}': Large score gap, keeping only top term "
                    f"'{cluster_terms_sorted[0].term}' (score={top_score:.3f})"
                )
                continue

            logger.info(
                f"Cluster '{node.category_name}': Score gap small, using LLM to filter "
                f"{len(cluster_terms)} terms"
            )

            try:
                llm = AzureChatOpenAI(
                    azure_endpoint=config.azure_openai_endpoint,
                    api_key=config.azure_openai_api_key,
                    api_version=config.azure_openai_api_version,
                    azure_deployment=llm_model,
                    temperature=0.0
                )

                prompt = ChatPromptTemplate.from_messages(
                    get_cluster_term_filtering_prompt_messages()
                )
                chain = prompt | llm | StrOutputParser()

                terms_info = "\n".join([
                    f"{i+1}. 「{t.term}」(スコア: {t.score:.3f})\n   定義: {t.definition or '（定義なし）'}"
                    for i, t in enumerate(cluster_terms_sorted)
                ])

                result_text = chain.invoke({
                    "category_name": node.category_name,
                    "terms_info": terms_info
                })

                result = self._parse_cluster_filtering_result(result_text)
                llm_call_count += 1

                if result and "keep_terms" in result:
                    keep_term_names = set(result["keep_terms"])
                    filtered_terms = [t for t in cluster_terms if t.term in keep_term_names]
                    terms_to_keep.extend(filtered_terms)
                    logger.info(
                        f"LLM filtered: {len(cluster_terms)} → {len(filtered_terms)} terms. "
                        f"Keeping: {keep_term_names}"
                    )
                else:
                    terms_to_keep.extend(cluster_terms_sorted[:1])
                    logger.warning(f"LLM filtering failed, keeping top term only")

            except Exception as e:
                logger.error(f"LLM filtering error: {e}, keeping top term only")
                terms_to_keep.append(cluster_terms_sorted[0])

        logger.info(
            f"Cluster-based filtering: {len(terms)} → {len(terms_to_keep)} terms "
            f"(LLM calls: {llm_call_count})"
        )

        return terms_to_keep

    def _parse_cluster_filtering_result(self, text: str) -> Optional[Dict]:
        """クラスタフィルタリング結果のJSONをパース"""
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
            logger.error(f"Failed to parse JSON: {text}")
            return None
