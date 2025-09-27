#!/usr/bin/env python3
"""
専門用語抽出システム V3 - 統合版
====================================
SemRe-Rank（統計的手法）+ RAGベクトル検索 + Azure OpenAI検証

機能:
- 統計的手法（TF-IDF、C-value）とグラフベース手法（kNN + Personalized PageRank）
- RAGベクトルストアからの類似文脈取得
- Azure OpenAIによるLLM検証
- PostgreSQLへの辞書保存
- SudachiPyによる高度な形態素解析
"""

import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import Counter, defaultdict
import math
import json
import pickle
import hashlib
import itertools
import logging
import re
from concurrent.futures import ThreadPoolExecutor

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# ベースクラスとデータモデルをインポート
from dictionary_system.core.models.base_extractor import BaseExtractor, Term

# 必要なライブラリ
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import unicodedata
from pydantic import BaseModel, Field

# 環境設定
from dotenv import load_dotenv
from rich.console import Console
import os

# データベース関連
from sqlalchemy import create_engine, text, inspect as sqlalchemy_inspect

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.documents import Document

# SudachiPy
try:
    from sudachipy import tokenizer, dictionary
    SUDACHI_AVAILABLE = True
except ImportError:
    SUDACHI_AVAILABLE = False
    print("Warning: SudachiPy not available. Using basic tokenization.")

# PyMuPDF
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

load_dotenv()
console = Console()

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# LangSmith設定の確認
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        logger.info(f"LangSmith tracing enabled - Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")

# ── Pydantic Models for Structured Output ────────
class TermStructured(BaseModel):
    """専門用語の構造"""
    headword: str = Field(description="専門用語の見出し語")
    synonyms: List[str] = Field(default_factory=list, description="類義語・別名のリスト")
    definition: str = Field(description="30-50字程度の簡潔な定義")

class TermListStructured(BaseModel):
    """用語リストの構造"""
    terms: List[TermStructured] = Field(default_factory=list, description="専門用語のリスト")

# ── Enhanced Term Extractor V3 ─────────────────
class EnhancedTermExtractorV3(BaseExtractor):
    """統合版専門用語抽出器（SemRe-Rank + RAG + Azure OpenAI）"""

    def __init__(
        self,
        # 既存パラメータ
        use_llm_validation: bool = True,
        min_term_length: int = 2,
        max_term_length: int = 10,
        min_frequency: int = 2,
        # SemRe-Rank用パラメータ
        k_neighbors: int = 12,
        sim_threshold: float = 0.30,
        alpha: float = 0.85,
        gamma: float = 0.7,
        beta: float = 0.3,
        w_pagerank: float = 0.6,
        embedding_model: Optional[str] = None,  # Azure OpenAI Embeddingsを使うためNone
        use_cache: bool = True,
        cache_dir: str = "cache/embeddings",
        # RAG/Azure OpenAI用パラメータ
        use_rag_context: bool = True,
        use_azure_openai: bool = True,  # Azure OpenAIを必須に
        db_url: Optional[str] = None,
        collection_name: str = "documents",
        jargon_table_name: str = "jargon_dictionary"
    ):
        """
        初期化

        Args:
            use_rag_context: RAGベクトル検索を使用するか
            use_azure_openai: Azure OpenAIを使用するか（Falseの場合は通常のOpenAI）
            db_url: PostgreSQL接続URL
            collection_name: ベクトルストアのコレクション名
            jargon_table_name: 用語辞書テーブル名
        """
        super().__init__()

        # 基本パラメータ
        self.use_llm_validation = use_llm_validation
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.min_frequency = min_frequency

        # SemRe-Rankパラメータ
        self.k_neighbors = k_neighbors
        self.sim_threshold = sim_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.w_pagerank = w_pagerank
        self.embedding_model = embedding_model or "paraphrase-multilingual-MiniLM-L12-v2"
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # 埋め込みモデルの初期化
        if not embedding_model:  # Azure OpenAI Embeddingsを使わない場合
            self.embedder = SentenceTransformer(self.embedding_model)

        # キャッシュディレクトリの作成
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_rag_context = use_rag_context
        self.use_azure_openai = use_azure_openai
        self.collection_name = collection_name
        self.jargon_table_name = jargon_table_name

        # Azure OpenAI設定
        if use_azure_openai:
            self._setup_azure_openai()

        # RAGベクトルストア設定
        if use_rag_context and db_url:
            self._setup_rag_store(db_url)

        # SudachiPy設定
        if SUDACHI_AVAILABLE:
            self.sudachi_tokenizer = dictionary.Dictionary().create()
            self.sudachi_mode = tokenizer.Tokenizer.SplitMode.C
            console.print("[cyan]SudachiPy形態素解析エンジンを初期化[/cyan]")

        # ドメイン特化辞書
        self._setup_domain_knowledge()

    def _setup_azure_openai(self):
        """Azure OpenAIの設定"""
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

        if not all([self.azure_endpoint, self.azure_api_key, self.azure_chat_deployment]):
            console.print("[yellow]Azure OpenAI設定が不完全。通常のOpenAIモードで動作します。[/yellow]")
            self.use_azure_openai = False
            return

        # Azure Embeddings (text-embedding-3-small)
        self.azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_deployment="text-embedding-3-small"  # 明示的に指定
        )

        # Azure Chat (gpt-4.1-miniを使用)
        self.azure_llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_deployment="gpt-4.1-mini",  # 明示的に指定
            temperature=0.1,
        )

        console.print("[green]Azure OpenAI設定完了[/green]")

    def _setup_rag_store(self, db_url: str):
        """RAGベクトルストアの設定"""
        try:
            self.vector_store = PGVector(
                collection_name=self.collection_name,
                connection_string=db_url,
                embedding_function=self.azure_embeddings if self.use_azure_openai else self.embedder,
                pre_delete_collection=False
            )
            self.db_engine = create_engine(db_url)
            console.print(f"[green]RAGベクトルストア接続完了: {self.collection_name}[/green]")
        except Exception as e:
            console.print(f"[yellow]RAGベクトルストア接続失敗: {e}[/yellow]")
            self.use_rag_context = False

    def extract_headings(self, text: str) -> Set[str]:
        """見出しから専門用語を抽出"""
        heading_terms = set()

        # 見出しパターン
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # マークダウン形式 # ## ### など
            r'^\d+\.\s+(.+)$',   # 1. 2. 3. など
            r'^[一二三四五六七八九十]+[、．]\s*(.+)$',  # 一、二、など
            r'^第[一二三四五六七八九十\d]+[章節項]\s*(.+)$',  # 第1章、第一節など
            r'^\[\d+\]\s*(.+)$',  # [1] [2] など
        ]

        # 行ごとに処理
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    heading_text = match.group(1)
                    # 見出しから用語候補を抽出
                    term_patterns = [
                        r'[ァ-ヶー]+',  # カタカナ
                        r'[一-龯]{2,}',  # 漢字（2文字以上）
                        r'[A-Z][A-Za-z0-9]*',  # 英数字
                        r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',  # カタカナ+漢字
                    ]

                    for term_pattern in term_patterns:
                        terms = re.findall(term_pattern, heading_text)
                        for term in terms:
                            if 2 <= len(term) <= 15:  # 最小/最大文字数チェック
                                heading_terms.add(term)
                    break

        return heading_terms

    def _is_likely_technical_term(self, term: str) -> bool:
        """技術用語らしさを判定"""
        # カタカナ+漢字の組み合わせ
        if re.search(r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+', term):
            return True
        # 英数字を含む
        if re.search(r'[A-Za-z0-9]', term):
            return True
        # 特定のサフィックス
        technical_suffixes = ['システム', '機関', '方式', '技術', '機構', 'エンジン', '装置', '設備']
        for suffix in technical_suffixes:
            if term.endswith(suffix):
                return True
        # 長い複合語（4文字以上）
        if len(term) >= 4:
            return True
        return False

    def _setup_domain_knowledge(self):
        """ストップワードの設定"""
        self.stopwords_extended = {
            "こと", "もの", "ため", "場合", "とき", "ところ", "方法",
            "状態", "結果", "目的", "対象", "内容", "情報", "データ",
            "システム", "プロセス", "サービス", "ソフトウェア",
            "確認", "実施", "作成", "使用", "管理", "処理", "記録",
            "年", "月", "日", "時", "分", "秒", "件", "個", "つ"
        }

    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """基本的な候補抽出（SudachiPy代替）"""
        candidates = defaultdict(int)

        # 正規表現パターンで日本語の複合語を抽出
        patterns = [
            r'[ァ-ヶー]+',  # カタカナ
            r'[一-龯]{2,}',  # 漢字（2文字以上）
            r'[A-Z][A-Za-z0-9]*',  # 英数字
            r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',  # カタカナ+漢字
            r'[一-龯]+[ァ-ヶー]+[一-龯]+',  # 漢字+カタカナ+漢字
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group()
                if self.min_term_length <= len(term) <= self.max_term_length:
                    candidates[term] += 1

        # 見出しから抽出した用語を追加
        heading_terms = self.extract_headings(text)
        for term in heading_terms:
            if term not in candidates:
                candidates[term] = 1
            candidates[term] += 2  # 見出しボーナス

        # 条件付きフィルタリング
        filtered_candidates = {}
        for term, freq in candidates.items():
            if freq >= 2:  # 通常の頻度条件
                filtered_candidates[term] = freq
            elif freq == 1 and (term in heading_terms or self._is_likely_technical_term(term)):
                # 頻度1でも見出しの用語または技術用語らしければ残す
                filtered_candidates[term] = freq

        return filtered_candidates

    def _calculate_tfidf(self, text: str, terms: List[str]) -> Dict[str, float]:
        """日本語対応のTF-IDF計算"""
        if not terms or not text:
            return {term: 0.0 for term in terms}

        # 各用語の出現頻度を計算
        doc_length = len(text)
        term_frequencies = {}

        for term in terms:
            count = text.count(term)
            term_frequencies[term] = count

        # 文書頻度（DF）を計算（簡易版：全文書を文単位で分割）
        sentences = text.replace('\n', '。').split('。')
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            # 文に分割できない場合は全体を１文書として扱う
            sentences = [text]

        doc_frequencies = {}
        for term in terms:
            doc_count = sum(1 for sent in sentences if term in sent)
            doc_frequencies[term] = max(doc_count, 1)  # ゼロ除算防止

        # TF-IDF計算
        tfidf_scores = {}
        total_docs = len(sentences)

        for term in terms:
            # TFにスムージングを追加（Laplace smoothing）
            tf = (term_frequencies[term] + 0.5) / (doc_length + 1.0)  # Term Frequency with smoothing
            # IDFの最小値を保証
            idf = max(0.1, math.log((total_docs + 1) / (doc_frequencies[term] + 1)) + 1)
            tfidf_scores[term] = tf * idf

        # スコアがすべて0の場合は頻度ベースにフォールバック
        if all(v == 0 for v in tfidf_scores.values()):
            for term in terms:
                tfidf_scores[term] = term_frequencies[term] / max(sum(term_frequencies.values()), 1)

        print(f"[DEBUG] TF-IDF計算結果: 非ゼロ項目={sum(1 for v in tfidf_scores.values() if v > 0)}/{len(terms)}")
        if tfidf_scores:
            print(f"[DEBUG] TF-IDF範囲: {min(tfidf_scores.values()):.6f} ~ {max(tfidf_scores.values()):.6f}")

        return tfidf_scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """スコアを正規化（最小値を0.01に設定して0を回避）"""
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {k: 0.5 for k in scores}  # 全て同じ値の場合

        # 最小値を0.01にして、完全な0を回避
        normalized = {}
        for k, v in scores.items():
            norm_value = (v - min_score) / (max_score - min_score)
            # 0.01～1.0の範囲にスケーリング
            normalized[k] = 0.01 + norm_value * 0.99

        return normalized

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """テキストの埋め込みベクトルを生成（統合版）"""
        if self.use_cache:
            cache_file = self.cache_dir / f"embeddings_{hashlib.md5(str(texts).encode()).hexdigest()}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Azure OpenAI Embeddingsを使用
        if self.use_azure_openai and hasattr(self, 'azure_embeddings'):
            try:
                embeddings = self.azure_embeddings.embed_documents(texts)
                embeddings = np.array(embeddings)
            except Exception as e:
                logger.warning(f"Azure Embeddings failed: {e}. Falling back to sentence-transformers")
                # フォールバックでsentence-transformersを使用
                if not hasattr(self, 'embedder'):
                    self.embedder = SentenceTransformer(self.embedding_model)
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
        else:
            # sentence-transformersを使用
            if not hasattr(self, 'embedder'):
                self.embedder = SentenceTransformer(self.embedding_model)
            embeddings = self.embedder.encode(terms, show_progress_bar=False)

        # キャッシュに保存
        if self.use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)

        return embeddings

    def _build_knn_graph(self, terms: List[Term], embeddings: np.ndarray) -> nx.Graph:
        """用語間のkNNグラフを構築"""
        n_terms = len(terms)
        if n_terms < 2:
            return nx.Graph()

        # k近傍探索
        k = min(self.k_neighbors, n_terms - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nbrs.fit(embeddings)

        distances, indices = nbrs.kneighbors(embeddings)

        # グラフ構築
        graph = nx.Graph()

        # ノード追加
        for i, term in enumerate(terms):
            graph.add_node(term.term, index=i, score=term.score)

        # エッジ追加
        for i in range(n_terms):
            for j, dist in zip(indices[i][1:], distances[i][1:]):  # 自分自身を除外
                similarity = 1.0 - dist  # コサイン距離から類似度へ
                if similarity >= self.sim_threshold:
                    graph.add_edge(
                        terms[i].term,
                        terms[j].term,
                        weight=float(similarity)
                    )

        return graph

    def _personalized_pagerank(self, graph: nx.Graph, terms: List[Term]) -> Dict[str, float]:
        """Personalized PageRank計算（正しいSemRe-Rank実装）"""
        if len(graph) == 0:
            return {}

        # base_scoreを初期重みとして設定
        personalization = {}
        for term in terms:
            if term.term in graph:
                # base_scoreのgamma乗を初期重みとする
                personalization[term.term] = max(term.score, 0.01) ** self.gamma

        # 正規化
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        else:
            # フォールバック：均等な重み
            personalization = {node: 1.0/len(graph) for node in graph.nodes()}

        # PageRankを一度だけ実行
        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=100,
                weight='weight',
                tol=1e-6
            )
        except:
            # エラー時のフォールバック
            pagerank_scores = {node: 1.0 / len(graph) for node in graph.nodes()}

        return pagerank_scores

    def _fuse_scores(self, terms: List[Term], pagerank_scores: Dict[str, float]) -> List[Term]:
        """スコアを統合"""
        # PageRankスコアを0.1～0.9の範囲に正規化（極端な値を避ける）
        if pagerank_scores:
            values = list(pagerank_scores.values())
            min_pr = min(values)
            max_pr = max(values)

            if max_pr > min_pr:
                # 0.1～0.9の範囲にマッピング
                pr_normalized = {
                    k: 0.1 + (v - min_pr) / (max_pr - min_pr) * 0.8
                    for k, v in pagerank_scores.items()
                }
            else:
                pr_normalized = {k: 0.5 for k in pagerank_scores}
        else:
            pr_normalized = {}

        for term in terms:
            # PageRankスコア取得（デフォルト0.5）
            pr_score = pr_normalized.get(term.term, 0.5)

            # スコア統合（TF-IDF + C-value + PageRank）
            term.score = term.score * (1.0 - self.w_pagerank) + pr_score * self.w_pagerank

            # PageRankスコアをメタデータに追加（表示用）
            term.metadata['pagerank'] = pr_score

        # 再度スコアでソート
        terms.sort(key=lambda x: x.score, reverse=True)
        return terms

    def _apply_cooccurrence_weight(self, graph: nx.Graph, text: str):
        """共起関係に基づくエッジ重みの調整"""
        sentences = text.replace('\n', '。').split('。')
        cooccur_counts = defaultdict(int)
        terms = list(graph.nodes())

        for sent in sentences:
            present_terms = []
            for term in terms:
                if term in sent:
                    present_terms.append(term)

            for t1, t2 in itertools.combinations(present_terms, 2):
                key = tuple(sorted([t1, t2]))
                cooccur_counts[key] += 1

        # エッジ重みを更新
        for u, v, data in graph.edges(data=True):
            key = tuple(sorted([u, v]))
            cooccur = cooccur_counts.get(key, 0)

            if cooccur > 0:
                factor = 1.0 + self.beta * np.log1p(cooccur)
                data['weight'] = float(data['weight'] * factor)

    def _extract_contexts(self, text: str, term: str, window_size: int = 50) -> List[str]:
        """用語の出現文脈を抽出"""
        contexts = []
        indices = [m.start() for m in re.finditer(re.escape(term), text)]

        for idx in indices[:3]:  # 最大3つの文脈
            start = max(0, idx - window_size)
            end = min(len(text), idx + len(term) + window_size)
            context = text[start:end]
            contexts.append(context)

        return contexts

    def _determine_validation_count_by_distribution(self, terms: List[Term]) -> int:
        """エルボー法でスコアの急激な変化点を検出して検証数を決定"""
        if not terms or len(terms) < 3:
            return len(terms) if terms else 0

        scores = [t.score for t in terms]

        # 方法1: エルボー法（スコア差分の変化率）
        score_diffs = []
        for i in range(1, len(scores)):
            diff = scores[i-1] - scores[i]
            score_diffs.append(diff)

        # 2階差分で変化率の変化を検出
        if len(score_diffs) > 1:
            second_diffs = []
            for i in range(1, len(score_diffs)):
                second_diff = score_diffs[i] - score_diffs[i-1]
                second_diffs.append(second_diff)

            # エルボーポイントを検出（最大の2階差分）
            if second_diffs:
                elbow_index = np.argmax(second_diffs) + 2
                elbow_count = min(len(terms), max(10, min(elbow_index, 100)))  # 10～100件の範囲

                console.print(f"    [dim]エルボー検出: {elbow_count}件目[/dim]")

        # 方法2: 累積寄与率による判定（85%まで）
        cumsum = np.cumsum(scores)
        if cumsum[-1] > 0:
            cumsum_normalized = cumsum / cumsum[-1]
            # 累積寄与率が85%に達する点
            threshold_index = np.argmax(cumsum_normalized >= 0.85) + 1
            contribution_count = min(len(terms), max(10, min(threshold_index, 100)))

            console.print(f"    [dim]累積寄与率85%: {contribution_count}件目[/dim]")
        else:
            contribution_count = min(30, len(terms))

        # 両方の方法の平均を取る
        if 'elbow_count' in locals() and 'contribution_count' in locals():
            final_count = int((elbow_count + contribution_count) / 2)
        elif 'elbow_count' in locals():
            final_count = elbow_count
        elif 'contribution_count' in locals():
            final_count = contribution_count
        else:
            final_count = min(30, len(terms))  # フォールバック

        final_count = min(final_count, len(terms))
        console.print(f"    [cyan]最終検証数: {final_count}件[/cyan]")
        return final_count

    async def _validate_with_llm(self, terms: List[Term], text: str) -> List[Term]:
        """基本のLLM検証（Azure以外）"""
        # Azure LLM検証へフォールバック
        return await self._validate_with_azure_llm(terms, text)

    async def extract_terms_with_validation(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDFファイルから専門用語を抽出し、LLM検証を行う

        Args:
            pdf_path: PDFファイルパス

        Returns:
            抽出された専門用語のリスト
        """
        # PDFからテキストを抽出
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            # フォールバック：テキストファイルとして読み込み
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        # テキストから用語抽出
        terms = await self.extract_terms_from_text(text)

        # 結果を辞書形式に変換
        results = []
        for i, term in enumerate(terms):
            # 最初の5件のデバッグ出力
            if i < 5:
                if i == 0:
                    print("\n[DEBUG] 最終スコアサンプル（最初の5件）:")
                print(f"  {i+1}. {term.term}:")
                print(f"    総合スコア: {term.score:.3f}")
                print(f"    TF-IDF: {term.metadata.get('tfidf', 0.0):.3f}")
                print(f"    C-value: {term.metadata.get('cvalue', 0.0):.3f}")
                print(f"    PageRank: {term.metadata.get('pagerank', 0.0):.3f}")
                print(f"    頻度: {term.metadata.get('frequency', 0)}")

            results.append({
                'term': term.term,
                'score': term.score,
                'frequency': term.metadata.get('frequency', 0),
                'definition': term.definition,
                'c_value': term.metadata.get('cvalue', 0.0),
                'tfidf': term.metadata.get('tfidf', 0.0),
                'pagerank': term.metadata.get('pagerank', 0.0),
                'synonyms': term.metadata.get('synonyms', [])
            })

        return results

    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """基本的な候補抽出（SudachiPy代替）"""
        candidates = defaultdict(int)

        # 正規表現パターンで日本語の複合語を抽出
        patterns = [
            r'[ァ-ヶー]+',  # カタカナ
            r'[一-龯]{2,}',  # 漢字（2文字以上）
            r'[A-Z][A-Za-z0-9]*',  # 英数字
            r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',  # カタカナ+漢字
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group()
                if self.min_term_length <= len(term) <= self.max_term_length:
                    candidates[term] += 1

        # 見出しから抽出した用語を追加
        heading_terms = self.extract_headings(text)
        for term in heading_terms:
            if term not in candidates:
                candidates[term] = 1
            candidates[term] += 2  # 見出しボーナス

        # 条件付きフィルタリング
        filtered_candidates = {}
        for term, freq in candidates.items():
            if freq >= 2:
                filtered_candidates[term] = freq
            elif freq == 1 and (term in heading_terms or self._is_likely_technical_term(term)):
                filtered_candidates[term] = freq

        return filtered_candidates

    # 重複した古い実装を削除

    async def search_similar_contexts(self, query_text: str, n_results: int = 3) -> str:
        """RAGベクトルストアから類似文脈を取得"""
        if not self.use_rag_context or not hasattr(self, 'vector_store'):
            return ""

        try:
            # クエリの最適化
            sentences = query_text.split('。')
            if len(sentences) > 3:
                query = '。'.join(sentences[:2] + [sentences[-1]])
            else:
                query = query_text[:1000]

            # 類似度検索
            results_with_scores = self.vector_store.similarity_search_with_score(
                query,
                k=n_results * 2
            )

            # 結果のフィルタリング
            related_contexts = []
            seen_contents = set()

            # 動的閾値
            if results_with_scores:
                top_scores = [score for _, score in results_with_scores[:3]]
                dynamic_threshold = sum(top_scores) / len(top_scores) * 0.7 if top_scores else 0.7
            else:
                dynamic_threshold = 0.7

            for doc, score in results_with_scores:
                if score < dynamic_threshold:
                    continue

                content_hash = hash(doc.page_content[:200])
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)

                related_contexts.append((doc.page_content[:500], score))

                if len(related_contexts) >= n_results:
                    break

            if related_contexts:
                return "\n\n".join([
                    f"[関連文脈 {i+1} (類似度: {score:.2f})]\n{content}"
                    for i, (content, score) in enumerate(related_contexts)
                ])

            return ""

        except Exception as e:
            logger.error(f"RAG検索エラー: {e}")
            return ""

    def extract_candidates_with_sudachi(self, text: str) -> Dict[str, int]:
        """SudachiPyを使った候補語抽出"""
        if not SUDACHI_AVAILABLE:
            # フォールバック: 親クラスのメソッドを使用
            # 代替実装を使用
            return self._extract_candidates(text)

        candidates = defaultdict(int)

        # 形態素解析
        tokens = self.sudachi_tokenizer.tokenize(text, self.sudachi_mode)

        # 名詞句の抽出
        current_phrase = []
        for token in tokens:
            pos = token.part_of_speech()[0]

            if pos in ['名詞', '接頭辞']:
                current_phrase.append(token.surface())
            else:
                if current_phrase:
                    # 名詞句を結合
                    phrase = ''.join(current_phrase)
                    if self.min_term_length <= len(phrase) <= self.max_term_length:
                        candidates[phrase] += 1

                    # 部分的な組み合わせも候補に
                    if len(current_phrase) > 2:
                        for i in range(len(current_phrase) - 1):
                            sub_phrase = ''.join(current_phrase[i:])
                            if self.min_term_length <= len(sub_phrase) <= self.max_term_length:
                                candidates[sub_phrase] += 1

                    current_phrase = []

        # 最後の句を処理
        if current_phrase:
            phrase = ''.join(current_phrase)
            if self.min_term_length <= len(phrase) <= self.max_term_length:
                candidates[phrase] += 1

        # 見出しから抽出した用語を追加
        heading_terms = self.extract_headings(text)
        for term in heading_terms:
            if term not in candidates:
                candidates[term] = 1  # 見出しの用語は最低でも頻度1
            # 見出しの用語には追加ボーナス
            candidates[term] += 2

        # 条件付きフィルタリング（頻度1でも重要な用語は残す）
        filtered_candidates = {}
        for term, freq in candidates.items():
            if freq >= 2:  # 通常の頻度条件
                filtered_candidates[term] = freq
            elif freq == 1 and (term in heading_terms or self._is_likely_technical_term(term)):
                # 頻度1でも見出しの用語または技術用語らしければ残す
                filtered_candidates[term] = freq

        return filtered_candidates


    def _calculate_advanced_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """改良版C-value計算"""
        c_values = {}

        for candidate, freq in candidates.items():
            length = len(candidate)

            # より長い候補語を探す
            longer_terms = []
            for other in candidates:
                if other != candidate and candidate in other:
                    longer_terms.append(other)

            # C値を計算
            if not longer_terms:
                c_value = math.log2(max(length, 2)) * freq
            else:
                sum_freq = sum(candidates[term] for term in longer_terms)
                t_a = len(longer_terms)
                c_value = math.log2(max(length, 2)) * (freq - sum_freq / t_a)

            c_values[candidate] = max(c_value, 0.0)

        return c_values


    async def extract_terms_from_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Term]:
        """
        統合版専門用語抽出

        Args:
            text: 抽出対象のテキスト
            metadata: 追加メタデータ

        Returns:
            抽出された専門用語リスト
        """
        console.print("[cyan]Enhanced V3で専門用語を抽出中...[/cyan]")

        # 1. SudachiPyで候補抽出
        console.print("  [yellow]1. 候補語抽出（SudachiPy）[/yellow]")
        if SUDACHI_AVAILABLE:
            candidates = self.extract_candidates_with_sudachi(text)
        else:
            candidates = self._extract_candidates(text)

        console.print(f"    [dim]候補数: {len(candidates)}[/dim]")

        # 2. 拡張スコア計算
        console.print("  [yellow]2. TF-IDFとC-value計算[/yellow]")
        # TF-IDFとC-valueを計算
        tfidf_scores = self._calculate_tfidf(text, list(candidates.keys()))
        cvalue_scores = self._calculate_advanced_cvalue(candidates)

        # スコアを正規化
        print(f"[DEBUG] TF-IDFスコアサンプル（最初の5件）:")
        for term in list(tfidf_scores.keys())[:5]:
            print(f"  {term}: {tfidf_scores[term]:.6f}")

        tfidf_normalized = self._normalize_scores(tfidf_scores)
        cvalue_normalized = self._normalize_scores(cvalue_scores)

        print(f"[DEBUG] 正規化後TF-IDFサンプル（最初の5件）:")
        for term in list(tfidf_normalized.keys())[:5]:
            print(f"  {term}: {tfidf_normalized[term]:.6f}")

        # 4. Termオブジェクト作成
        terms = []
        for term, frequency in candidates.items():
            if frequency < self.min_frequency:
                continue

            # 統合スコア（TF-IDFとC-valueのみ）
            tfidf = tfidf_normalized.get(term, 0.0)
            cvalue = cvalue_normalized.get(term, 0.0)
            base_score = (tfidf * 0.4 + cvalue * 0.6)

            # ストップワードチェック
            if term in self.stopwords_extended:
                base_score *= 0.1

            terms.append(Term(
                term=term,
                score=base_score,
                definition="",
                metadata={
                    "tfidf": tfidf,
                    "cvalue": cvalue,
                    "base_score": base_score,
                    "frequency": frequency,
                    "contexts": self._extract_contexts(text, term)
                }
            ))

        # スコアでソート
        terms.sort(key=lambda x: x.score, reverse=True)

        # 5. 前処理フィルタ
        console.print(f"  [yellow]3. 前処理フィルタ（候補数: {len(terms)}）[/yellow]")
        terms = self._prefilter_terms_extended(terms)
        console.print(f"    [dim]フィルタ後: {len(terms)}件[/dim]")

        # 6. グラフ構築用に上位N件に絞り込み
        max_graph_nodes = min(300, len(terms))
        if len(terms) > max_graph_nodes:
            terms = terms[:max_graph_nodes]
            console.print(f"    [dim]グラフ構築用に上位{max_graph_nodes}件に絞り込み[/dim]")

        if len(terms) == 0:
            return []

        # 7. 埋め込みを計算（用語＋文脈）
        console.print("  [yellow]4. 埋め込みを計算（文脈付き）[/yellow]")
        # 用語と文脈を結合してより精度の高い埋め込みを生成
        texts = []
        for t in terms:
            contexts = t.metadata.get('contexts', [])
            if contexts and contexts[0]:
                # 最初の文脈の一部を追加（50文字まで）
                context_snippet = contexts[0][:50]
                text = f"{t.term} [SEP] {context_snippet}"
            else:
                text = t.term
            texts.append(text)

        embeddings = self._compute_embeddings(texts)

        # 8. kNNグラフを構築
        console.print("  [yellow]5. kNNグラフを構築[/yellow]")
        graph = self._build_knn_graph(terms, embeddings)

        # 9. RAG文脈で共起を強化
        if self.use_rag_context:
            console.print("  [yellow]6. RAG文脈で共起を強化[/yellow]")
            await self._enhance_with_rag_context(graph, text)
        else:
            # 通常の共起補正
            self._apply_cooccurrence_weight(graph, text)

        # 10. Personalized PageRankを実行
        console.print("  [yellow]7. Personalized PageRankを実行[/yellow]")
        pagerank_scores = self._personalized_pagerank(graph, terms)

        # 11. スコアを統合
        console.print("  [yellow]8. スコアを統合[/yellow]")
        terms = self._fuse_scores(terms, pagerank_scores)

        # 12. 検証数を決定
        validation_count = self._determine_validation_count_by_distribution(terms)

        # 12.5. MMRで多様な候補を選択（新規追加）
        if len(terms) > validation_count:
            # embeddings変数は935-950行目で計算済み
            terms_selected = self._mmr_select(
                terms=terms,
                embeddings=embeddings,  # 既に正規化済みの埋め込み
                k=validation_count,
                lambda_param=0.7,  # 関連性重視
                use_embedding=True  # 埋め込みベース推奨
            )
            console.print(f"  [green]MMRで{len(terms)}件から多様な{len(terms_selected)}件を選択[/green]")
        else:
            terms_selected = terms[:validation_count]

        # 13. LLM検証（MMR選択後の用語のみ）
        if self.use_llm_validation and terms_selected:
            console.print(f"  [cyan]LLM検証対象: {len(terms_selected)}件（MMR適用済み）[/cyan]")
            if self.use_azure_openai:
                terms_final = await self._validate_with_azure_llm(terms_selected, text)
            else:
                terms_final = await self._validate_with_llm(terms_selected, text)
            console.print(f"  [yellow]LLM検証後の最終候補数: {len(terms_final)}[/yellow]")
        else:
            terms_final = terms_selected

        # 14. データベース保存（オプション）
        if hasattr(self, 'db_engine'):
            await self._save_to_database(terms_final)

        # 最終結果
        final_terms = terms_final[:min(validation_count, 50)]
        console.print(f"  [green]最終的な専門用語数: {len(final_terms)}[/green]")
        return final_terms

    def _prefilter_terms_extended(self, terms: List[Term]) -> List[Term]:
        """拡張版前処理フィルタ（包含関係フィルタを追加）"""
        filtered = []

        for term in terms:
            norm_term = unicodedata.normalize('NFKC', term.term)

            # ストップワードチェック
            if norm_term in self.stopwords_extended:
                continue

            # 1文字の語は除外
            if len(norm_term) == 1:
                continue

            # 数値のみは除外
            if norm_term.isdigit():
                continue

            # 年号パターン除外（20XX, 19XXなど）
            if norm_term.startswith(('19', '20')) and len(norm_term) == 4:
                try:
                    int(norm_term)
                    continue
                except ValueError:
                    pass

            filtered.append(term)

        # 包含関係フィルタ：他の用語に含まれる単独語を減点
        for term in filtered:
            if len(term.term) <= 3:  # 短い用語のみチェック
                # 他の長い用語に含まれているかチェック
                is_contained = False
                for other in filtered:
                    if term.term != other.term and term.term in other.term and len(other.term) > len(term.term):
                        is_contained = True
                        break

                if is_contained:
                    # 包含されている場合、スコアを減点（削除ではなく減点）
                    term.score *= 0.3
                    term.metadata['contained'] = True

        return filtered

    async def _enhance_with_rag_context(self, graph: nx.Graph, text: str):
        """RAG文脈情報でグラフを強化"""
        # RAG検索で関連文脈を取得
        related_contexts = await self.search_similar_contexts(text)

        if not related_contexts:
            # 通常の共起補正にフォールバック
            self._apply_cooccurrence_weight(graph, text)
            return

        # 関連文脈での共起をカウント
        cooccur_counts = defaultdict(int)
        terms = list(graph.nodes())

        # 元のテキストと関連文脈を結合
        combined_text = text + "\n" + related_contexts
        sentences = combined_text.replace('\n', '。').split('。')

        for sent in sentences:
            present_terms = []
            for term in terms:
                if term in sent:
                    present_terms.append(term)

            for t1, t2 in itertools.combinations(present_terms, 2):
                key = tuple(sorted([t1, t2]))
                cooccur_counts[key] += 1

        # エッジ重みを更新
        updated = 0
        for u, v, data in graph.edges(data=True):
            key = tuple(sorted([u, v]))
            cooccur = cooccur_counts.get(key, 0)

            if cooccur > 0:
                # RAG文脈での共起は重みを増加
                factor = 1.0 + self.beta * np.log1p(cooccur) * 1.5
                data['weight'] = float(data['weight'] * factor)
                updated += 1

        console.print(f"    [dim]RAG文脈で{updated}エッジを強化[/dim]")

    async def _validate_with_azure_llm(
        self,
        terms: List[Term],
        text: str
    ) -> List[Term]:
        """Azure OpenAIでLLM検証"""
        if not hasattr(self, 'azure_llm'):
            # フォールバック
            return await self._validate_with_llm(terms, text)

        # JSON出力パーサー
        json_parser = JsonOutputParser(pydantic_object=TermListStructured)

        # プロンプト
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門分野の用語抽出専門家です。
与えられた候補から真の専門用語のみを厳密に選定してください。

【選定基準】
1. ドメイン固有性：その分野特有の概念
2. 定義の必要性：一般人には説明が必要
3. 専門的価値：その分野で重要な意味を持つ

【出力内容】
各専門用語について：
- headword: 専門用語の見出し語
- synonyms: 類義語、略語、別名のリスト（文脈から推測できるもの、一般的に知られているもの）
- definition: 30-50字程度の簡潔な定義

{format_instructions}"""),
            ("user", """以下の候補から専門用語を選定し、類義語も含めて抽出してください：

文脈:
{text}

候補リスト:
{candidates}

各候補について判定し、類義語や略語も抽出してJSON形式で出力してください。
例えば「アンモニア燃料」の場合、「NH3燃料」「アンモニア系燃料」などが類義語になります。""")
        ]).partial(format_instructions=json_parser.get_format_instructions())

        # チェイン構築（LCEL）
        chain = validation_prompt | self.azure_llm | json_parser

        try:
            # 候補を文字列化
            candidates_str = "\n".join([
                f"- {t.term} (頻度: {t.metadata.get('frequency', 0)}, スコア: {t.score:.3f})"
                for t in terms
            ])

            # LLM実行
            result = await chain.ainvoke({
                "text": text[:3000],  # テキストを制限
                "candidates": candidates_str
            })

            # 結果をTermオブジェクトに変換
            validated_terms = []

            # resultが辞書の場合と、オブジェクトの場合の両方に対応
            if isinstance(result, dict):
                # 辞書形式の場合
                terms_list = result.get('terms', [])
            else:
                # Pydanticオブジェクトの場合
                terms_list = result.terms if hasattr(result, 'terms') else []

            validated_set = {t.get('headword', t['headword']) if isinstance(t, dict) else t.headword for t in terms_list}

            for term in terms:
                if term.term in validated_set:
                    # メタデータ更新
                    for t in terms_list:
                        if isinstance(t, dict):
                            headword = t.get('headword', '')
                            definition = t.get('definition', '')
                            synonyms = t.get('synonyms', [])
                        else:
                            headword = t.headword
                            definition = t.definition if hasattr(t, 'definition') else ''
                            synonyms = t.synonyms if hasattr(t, 'synonyms') else []

                        if headword == term.term:
                            term.definition = definition
                            term.metadata["synonyms"] = synonyms if synonyms else []
                            break
                    validated_terms.append(term)

            return validated_terms

        except Exception as e:
            logger.error(f"Azure LLM検証エラー: {e}")
            return terms[:20]  # エラー時は上位20件を返す

    async def _save_to_database(self, terms: List[Term]):
        """データベースに保存"""
        if not hasattr(self, 'db_engine'):
            return

        try:
            # テーブル作成（存在しない場合）
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.jargon_table_name} (
                term VARCHAR(255) PRIMARY KEY,
                definition TEXT,
                synonyms TEXT,
                score FLOAT,
                frequency INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            with self.db_engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()

                # 既存の用語を取得
                existing_terms = set()
                result = conn.execute(text(f"SELECT term FROM {self.jargon_table_name}"))
                for row in result:
                    existing_terms.add(row[0])

                # 新規用語を挿入
                new_terms = []
                for term in terms:
                    if term.term not in existing_terms:
                        synonyms = json.dumps(
                            term.metadata.get("synonyms", []),
                            ensure_ascii=False
                        )
                        new_terms.append({
                            "term": term.term,
                            "definition": term.definition,
                            "synonyms": synonyms,
                            "score": term.score,
                            "frequency": term.frequency
                        })

                if new_terms:
                    insert_sql = f"""
                    INSERT INTO {self.jargon_table_name}
                    (term, definition, synonyms, score, frequency)
                    VALUES (:term, :definition, :synonyms, :score, :frequency)
                    """
                    conn.execute(text(insert_sql), new_terms)
                    conn.commit()
                    console.print(f"    [green]{len(new_terms)}件の新規用語をDBに保存[/green]")

        except Exception as e:
            logger.error(f"DB保存エラー: {e}")

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """テキストの埋め込みベクトルを生成（統合版）"""
        if self.use_cache:
            cache_file = self.cache_dir / f"embeddings_{hashlib.md5(str(texts).encode()).hexdigest()}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Azure OpenAI Embeddingsを使用
        if self.use_azure_openai and hasattr(self, 'azure_embeddings'):
            try:
                embeddings = self.azure_embeddings.embed_documents(texts)
                embeddings = np.array(embeddings)
            except Exception as e:
                logger.warning(f"Azure Embeddings failed: {e}. Falling back to sentence-transformers")
                # フォールバックでsentence-transformersを使用
                if not hasattr(self, 'embedder'):
                    self.embedder = SentenceTransformer(self.embedding_model)
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
        else:
            # sentence-transformersを使用
            if not hasattr(self, 'embedder'):
                self.embedder = SentenceTransformer(self.embedding_model)
            embeddings = self.embedder.encode(terms, show_progress_bar=False)

        # キャッシュに保存
        if self.use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)

        return embeddings

    def _build_knn_graph(self, terms: List[Term], embeddings: np.ndarray) -> nx.Graph:
        """用語間のkNNグラフを構築"""
        n_terms = len(terms)
        if n_terms < 2:
            return nx.Graph()

        # k近傍探索
        k = min(self.k_neighbors, n_terms - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nbrs.fit(embeddings)

        distances, indices = nbrs.kneighbors(embeddings)

        # グラフ構築
        graph = nx.Graph()

        # ノード追加
        for i, term in enumerate(terms):
            graph.add_node(term.term, index=i, score=term.score)

        # エッジ追加
        for i in range(n_terms):
            for j, dist in zip(indices[i][1:], distances[i][1:]):  # 自分自身を除外
                similarity = 1.0 - dist  # コサイン距離から類似度へ
                if similarity >= self.sim_threshold:
                    graph.add_edge(
                        terms[i].term,
                        terms[j].term,
                        weight=float(similarity)
                    )

        return graph

    def _personalized_pagerank(self, graph: nx.Graph, terms: List[Term]) -> Dict[str, float]:
        """Personalized PageRank計算（正しいSemRe-Rank実装）"""
        if len(graph) == 0:
            return {}

        # base_scoreを初期重みとして設定
        personalization = {}
        for term in terms:
            if term.term in graph:
                # base_scoreのgamma乗を初期重みとする
                personalization[term.term] = max(term.score, 0.01) ** self.gamma

        # 正規化
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        else:
            # フォールバック：均等な重み
            personalization = {node: 1.0/len(graph) for node in graph.nodes()}

        # PageRankを一度だけ実行
        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=100,
                weight='weight',
                tol=1e-6
            )
        except:
            # エラー時のフォールバック
            pagerank_scores = {node: 1.0 / len(graph) for node in graph.nodes()}

        return pagerank_scores

    def _fuse_scores(self, terms: List[Term], pagerank_scores: Dict[str, float]) -> List[Term]:
        """スコアを統合"""
        # PageRankスコアを0.1～0.9の範囲に正規化（極端な値を避ける）
        if pagerank_scores:
            values = list(pagerank_scores.values())
            min_pr = min(values)
            max_pr = max(values)

            if max_pr > min_pr:
                # 0.1～0.9の範囲にマッピング
                pr_normalized = {
                    k: 0.1 + (v - min_pr) / (max_pr - min_pr) * 0.8
                    for k, v in pagerank_scores.items()
                }
            else:
                pr_normalized = {k: 0.5 for k in pagerank_scores}
        else:
            pr_normalized = {}

        for term in terms:
            # PageRankスコア取得（デフォルト0.5）
            pr_score = pr_normalized.get(term.term, 0.5)

            # スコア統合（TF-IDF + C-value + PageRank）
            term.score = term.score * (1.0 - self.w_pagerank) + pr_score * self.w_pagerank

            # PageRankスコアをメタデータに追加（表示用）
            term.metadata['pagerank'] = pr_score

        # 再度スコアでソート
        terms.sort(key=lambda x: x.score, reverse=True)
        return terms

    def _apply_cooccurrence_weight(self, graph: nx.Graph, text: str):
        """共起関係に基づくエッジ重みの調整"""
        sentences = text.replace('\n', '。').split('。')
        cooccur_counts = defaultdict(int)
        terms = list(graph.nodes())

        for sent in sentences:
            present_terms = []
            for term in terms:
                if term in sent:
                    present_terms.append(term)

            for t1, t2 in itertools.combinations(present_terms, 2):
                key = tuple(sorted([t1, t2]))
                cooccur_counts[key] += 1

        # エッジ重みを更新
        for u, v, data in graph.edges(data=True):
            key = tuple(sorted([u, v]))
            cooccur = cooccur_counts.get(key, 0)

            if cooccur > 0:
                factor = 1.0 + self.beta * np.log1p(cooccur)
                data['weight'] = float(data['weight'] * factor)

    def _extract_contexts(self, text: str, term: str, window_size: int = 50) -> List[str]:
        """用語の出現文脈を抽出"""
        contexts = []
        indices = [m.start() for m in re.finditer(re.escape(term), text)]

        for idx in indices[:3]:  # 最大3つの文脈
            start = max(0, idx - window_size)
            end = min(len(text), idx + len(term) + window_size)
            context = text[start:end]
            contexts.append(context)

        return contexts

    def _compute_jaccard_similarity(self, term1: str, term2: str, n: int = 2) -> float:
        """
        n-gramベースのJaccard係数を計算

        Args:
            term1, term2: 比較する用語
            n: n-gramのn（デフォルト2=bigram）

        Returns:
            類似度（0-1）
        """
        if len(term1) < n or len(term2) < n:
            return 1.0 if term1 == term2 else 0.0

        # n-gram生成
        ngrams1 = {term1[i:i+n] for i in range(len(term1)-n+1)}
        ngrams2 = {term2[i:i+n] for i in range(len(term2)-n+1)}

        # Jaccard係数
        if not ngrams1 or not ngrams2:
            return 0.0
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0

    def _mmr_select(
        self,
        terms: List[Term],
        embeddings: Optional[np.ndarray],
        k: int,
        lambda_param: float = 0.7,
        use_embedding: bool = True
    ) -> List[Term]:
        """
        Maximal Marginal Relevanceによる多様性を考慮した選択

        Args:
            terms: 候補用語リスト（スコア順）
            embeddings: 埋め込みベクトル（既に計算済みを再利用）
            k: 選択数（エルボー法で決定済み）
            lambda_param: 関連性と多様性のバランス（0.7=関連性重視）
            use_embedding: True=埋め込み類似度、False=文字列類似度

        Returns:
            多様性を持つk個の用語
        """
        if len(terms) <= k:
            return terms

        selected = []
        selected_indices = []
        remaining = list(range(len(terms)))

        # 最初は最高スコアを選択
        selected.append(terms[0])
        selected_indices.append(0)
        remaining.remove(0)

        # 残りをMMRで選択
        while len(selected) < k and remaining:
            best_mmr = -float('inf')
            best_idx = -1

            for idx in remaining:
                # 関連性スコア（正規化済み）
                relevance = terms[idx].score

                # 既選択との最大類似度
                max_sim = 0.0
                for sel_idx in selected_indices:
                    if use_embedding and embeddings is not None:
                        # 埋め込みベースの類似度（コサイン）
                        sim = float(np.dot(embeddings[idx], embeddings[sel_idx]))
                    else:
                        # 文字列ベースの類似度（Jaccard）
                        sim = self._compute_jaccard_similarity(
                            terms[idx].term,
                            terms[sel_idx].term
                        )
                    max_sim = max(max_sim, sim)

                # MMRスコア計算
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0:
                selected.append(terms[best_idx])
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return selected

    async def _validate_with_llm(self, terms: List[Term], text: str) -> List[Term]:
        """基本のLLM検証（Azure以外）"""
        # Azure LLM検証へフォールバック
        return await self._validate_with_azure_llm(terms, text)

    async def extract_terms_with_validation(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDFファイルから専門用語を抽出し、LLM検証を行う

        Args:
            pdf_path: PDFファイルパス

        Returns:
            抽出された専門用語のリスト
        """
        # PDFからテキストを抽出
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            # フォールバック：テキストファイルとして読み込み
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        # テキストから用語抽出
        terms = await self.extract_terms_from_text(text)

        # 結果を辞書形式に変換
        results = []
        for i, term in enumerate(terms):
            # 最初の5件のデバッグ出力
            if i < 5:
                if i == 0:
                    print("\n[DEBUG] 最終スコアサンプル（最初の5件）:")
                print(f"  {i+1}. {term.term}:")
                print(f"    総合スコア: {term.score:.3f}")
                print(f"    TF-IDF: {term.metadata.get('tfidf', 0.0):.3f}")
                print(f"    C-value: {term.metadata.get('cvalue', 0.0):.3f}")
                print(f"    PageRank: {term.metadata.get('pagerank', 0.0):.3f}")
                print(f"    頻度: {term.metadata.get('frequency', 0)}")

            results.append({
                'term': term.term,
                'score': term.score,
                'frequency': term.metadata.get('frequency', 0),
                'definition': term.definition,
                'c_value': term.metadata.get('cvalue', 0.0),
                'tfidf': term.metadata.get('tfidf', 0.0),
                'pagerank': term.metadata.get('pagerank', 0.0),
                'synonyms': term.metadata.get('synonyms', [])
            })

        return results

    def extract(self, text: str, **kwargs) -> List[Term]:
        """
        BaseExtractorの抽象メソッドの実装

        Args:
            text: 抽出対象テキスト
            **kwargs: 追加パラメータ

        Returns:
            抽出された専門用語のList[Term]
        """
        # 同期的に非同期関数を実行
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            terms = loop.run_until_complete(self.extract_terms_from_text(text, metadata=kwargs))
            return terms
        finally:
            loop.close()

async def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        console.print("[red]使用法: python term_extractor_v3.py <input_path> [output.json][/red]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "output/enhanced_v3_terms.json")

    # DB URL取得
    db_url = os.getenv("DATABASE_URL") or os.getenv("PG_URL")

    # 抽出器を初期化
    extractor = EnhancedTermExtractorV3(
        use_llm_validation=True,
        min_frequency=2,
        k_neighbors=12,
        sim_threshold=0.30,
        w_pagerank=0.6,
        use_rag_context=True,
        use_azure_openai=True,
        db_url=db_url
    )

    # 処理実行
    if input_path.is_file():
        result = await extractor.extract_from_file(input_path)
        results = [result]
    else:
        results = await extractor.extract_from_directory(input_path)

    if results:
        # 結果を表示
        all_terms = []
        for result in results:
            all_terms.extend(result.terms)

        merged_terms = extractor.merge_terms([all_terms])
        extractor.display_terms(merged_terms)

        # 保存
        extractor.save_results(results, output_path)

        # 統計を表示
        stats = extractor.get_statistics(results)
        console.print(f"\n[green]処理統計:[/green]")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())