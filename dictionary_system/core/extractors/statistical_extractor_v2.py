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
from concurrent.futures import ThreadPoolExecutor

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# 既存のベースクラスを継承
from extractors.statistical_extractor_V2 import StatisticalTermExtractorV2
from extractors.statistical_extractor import StatisticalTermExtractor
from src.utils.base_extractor import Term

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
class EnhancedTermExtractorV3(StatisticalTermExtractorV2):
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
        use_azure_openai: bool = False,
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
        super().__init__(
            use_llm_validation=use_llm_validation,
            min_term_length=min_term_length,
            max_term_length=max_term_length,
            min_frequency=min_frequency,
            k_neighbors=k_neighbors,
            sim_threshold=sim_threshold,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            w_pagerank=w_pagerank,
            embedding_model=embedding_model,
            use_cache=use_cache,
            cache_dir=cache_dir
        )

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

    def _setup_domain_knowledge(self):
        """ドメイン知識の設定"""
        self.domain_keywords = {
            "医薬": ["品", "部外品", "製剤", "原薬", "添加剤", "成分", "薬効", "薬理", "薬物"],
            "製造": ["管理", "工程", "バリデーション", "設備", "施設", "製法", "品質", "検証"],
            "品質": ["管理", "保証", "試験", "規格", "基準", "標準", "適合", "検査"],
            "規制": ["要件", "申請", "承認", "届出", "査察", "法令", "通知", "ガイドライン"],
            "安全": ["性", "評価", "リスク", "毒性", "副作用", "有害", "事象"],
            "技術": ["分析", "方法", "手法", "システム", "プロセス", "機器", "装置"],
        }

        self.stopwords_extended = {
            "こと", "もの", "ため", "場合", "とき", "ところ", "方法",
            "状態", "結果", "目的", "対象", "内容", "情報", "データ",
            "システム", "プロセス", "サービス", "ソフトウェア",
            "確認", "実施", "作成", "使用", "管理", "処理", "記録",
            "年", "月", "日", "時", "分", "秒", "件", "個", "つ"
        }

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

        return dict(candidates)

    def calculate_enhanced_scores(self, candidates: Dict[str, int], text: str) -> Dict[str, float]:
        """拡張版スコア計算（C-value、分布、位置、共起）"""
        scores = {}
        doc_length = len(text)

        # C-value計算
        c_values = self._calculate_advanced_cvalue(candidates)

        for candidate, freq in candidates.items():
            # 出現位置を収集
            positions = []
            start = 0
            while True:
                pos = text.find(candidate, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1

            # 各種スコア計算
            c_value = c_values.get(candidate, 0.0)
            distribution_score = self._calculate_distribution_score(positions, doc_length)
            position_score = self._calculate_position_score(positions[0] if positions else doc_length, doc_length)
            domain_score = self._calculate_domain_score(candidate)

            # 統合スコア
            scores[candidate] = (
                c_value * 0.4 +
                distribution_score * 0.2 +
                position_score * 0.1 +
                domain_score * 0.3
            )

        return scores

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

    def _calculate_distribution_score(self, positions: List[int], doc_length: int) -> float:
        """文書内分布スコア"""
        if len(positions) <= 1:
            return 0.5

        mean_pos = sum(positions) / len(positions)
        variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
        std_dev = math.sqrt(variance)

        ideal_gap = doc_length / (len(positions) + 1)
        distribution_score = 1.0 / (1.0 + std_dev / (ideal_gap + 1))

        return distribution_score

    def _calculate_position_score(self, first_pos: int, doc_length: int) -> float:
        """初出位置スコア"""
        return 1.0 - (first_pos / doc_length) * 0.5

    def _calculate_domain_score(self, candidate: str) -> float:
        """ドメイン関連スコア"""
        score = 0.0
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in candidate:
                    score += 0.1
        return min(score, 1.0)

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
        console.print("  [yellow]2. 拡張スコア計算[/yellow]")
        enhanced_scores = self.calculate_enhanced_scores(candidates, text)

        # 3. TF-IDFとC-valueを計算（親クラスのメソッド）
        tfidf_scores = self._calculate_tfidf(text, list(candidates.keys()))
        cvalue_scores = self._calculate_cvalue(candidates)

        # スコアを正規化
        tfidf_normalized = self._normalize_scores(tfidf_scores)
        cvalue_normalized = self._normalize_scores(cvalue_scores)
        enhanced_normalized = self._normalize_scores(enhanced_scores)

        # 4. Termオブジェクト作成
        terms = []
        for term, frequency in candidates.items():
            if frequency < self.min_frequency:
                continue

            # 統合スコア
            tfidf = tfidf_normalized.get(term, 0.0)
            cvalue = cvalue_normalized.get(term, 0.0)
            enhanced = enhanced_normalized.get(term, 0.0)
            base_score = (tfidf * 0.2 + cvalue * 0.3 + enhanced * 0.5)

            # ストップワードチェック
            if term in self.stopwords_extended:
                base_score *= 0.1

            terms.append(Term(
                term=term,
                definition="",
                score=base_score,
                frequency=frequency,
                contexts=self._extract_contexts(text, term),
                metadata={
                    "tfidf": tfidf,
                    "cvalue": cvalue,
                    "enhanced": enhanced,
                    "base_score": base_score
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

        # 7. 埋め込みを計算
        console.print("  [yellow]4. 埋め込みを計算[/yellow]")
        embeddings = self._compute_embeddings([t.term for t in terms])

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

        # 13. LLM検証
        if self.use_llm_validation and terms:
            console.print(f"  [cyan]LLM検証対象: 上位{validation_count}件[/cyan]")
            if self.use_azure_openai:
                terms = await self._validate_with_azure_llm(terms[:validation_count], text)
            else:
                terms = await self._validate_with_llm(terms[:validation_count], text)
            console.print(f"  [yellow]LLM検証後の候補数: {len(terms)}[/yellow]")

        # 14. データベース保存（オプション）
        if hasattr(self, 'db_engine'):
            await self._save_to_database(terms)

        # 最終結果
        final_terms = terms[:min(validation_count, 50)]
        console.print(f"  [green]最終的な専門用語数: {len(final_terms)}[/green]")
        return final_terms

    def _prefilter_terms_extended(self, terms: List[Term]) -> List[Term]:
        """拡張版前処理フィルタ"""
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

{format_instructions}"""),
            ("user", """以下の候補から専門用語を選定してください：

文脈:
{text}

候補リスト:
{candidates}

各候補について判定し、JSON形式で出力してください。""")
        ]).partial(format_instructions=json_parser.get_format_instructions())

        # チェイン構築（LCEL）
        chain = validation_prompt | self.azure_llm | json_parser

        try:
            # 候補を文字列化
            candidates_str = "\n".join([
                f"- {t.term} (頻度: {t.frequency}, スコア: {t.score:.3f})"
                for t in terms
            ])

            # LLM実行
            result = await chain.ainvoke({
                "text": text[:3000],  # テキストを制限
                "candidates": candidates_str
            })

            # 結果をTermオブジェクトに変換
            validated_terms = []
            validated_set = {t.headword for t in result.terms}

            for term in terms:
                if term.term in validated_set:
                    # メタデータ更新
                    for t in result.terms:
                        if t.headword == term.term:
                            term.definition = t.definition
                            term.metadata["synonyms"] = t.synonyms
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