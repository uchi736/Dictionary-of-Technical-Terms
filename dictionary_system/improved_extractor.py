#!/usr/bin/env python3
"""
改良版専門用語抽出システム（スタンドアロン版）
- 形態素解析による複合語抽出
- TF-IDFとC-value統計
- 埋め込みベクトルによる類似度計算
- kNNグラフ + PageRankによるランキング
"""

import re
import json
import math
import hashlib
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Set

import fitz  # PyMuPDF
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


class ImprovedTermExtractor:
    """改良版専門用語抽出器"""

    def __init__(
        self,
        min_term_length: int = 2,
        max_term_length: int = 15,
        min_frequency: int = 2,
        k_neighbors: int = 10,
        sim_threshold: float = 0.35,
        alpha: float = 0.85,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        use_cache: bool = True,
        cache_dir: str = "cache/embeddings"
    ):
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.min_frequency = min_frequency
        self.k_neighbors = k_neighbors
        self.sim_threshold = sim_threshold
        self.alpha = alpha
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # 埋め込みモデル
        console.print(f"[cyan]埋め込みモデルを読み込み中: {embedding_model}[/cyan]")
        self.embedder = SentenceTransformer(embedding_model)

        # キャッシュディレクトリ
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ストップワード
        self.stop_words = {'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'ここ', 'そこ',
                           'あそこ', 'こちら', 'どこ', 'だれ', 'なに', 'なん', 'もの', 'こと'}

    def extract_text_from_pdf(self, pdf_path) -> str:
        """PDFからテキストを抽出"""
        console.print(f"[yellow]PDFを読み込み中: {pdf_path}[/yellow]")
        doc = fitz.open(pdf_path)
        text = ""
        total_pages = len(doc)

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            console.print(f"  [dim]ページ {page_num + 1}/{total_pages} 処理中[/dim]", end="\r")

        doc.close()
        console.print(f"  [green]全{total_pages}ページ読み込み完了[/green]")
        return text

    def extract_candidates_improved(self, text: str) -> Dict[str, int]:
        """改良版候補語抽出（複合語対応）"""
        # 日本語の複合名詞パターン
        patterns = [
            # カタカナ語
            (r'[ァ-ヶー]+', 'katakana'),
            # 漢字＋カタカナの複合語
            (r'[一-龯]+[ァ-ヶー]+', 'kanji_katakana'),
            # カタカナ＋漢字の複合語
            (r'[ァ-ヶー]+[一-龯]+', 'katakana_kanji'),
            # 漢字の連続（2文字以上）
            (r'[一-龯]{2,}', 'kanji'),
            # 英数字の略語
            (r'[A-Z]{2,}[0-9]*', 'acronym'),
            # 英単語
            (r'[A-Za-z]+', 'english'),
        ]

        # 複合語パターン（助詞でつながる専門用語）
        compound_patterns = [
            r'(\S+?)の(\S+)',  # 「〜の〜」
            r'(\S+?)および(\S+)',  # 「〜および〜」
            r'(\S+?)・(\S+)',  # 「〜・〜」
        ]

        candidates = Counter()

        # 文ごとに処理
        sentences = re.split(r'[。．！？\n]', text)

        for sentence in sentences:
            # 基本パターンでの抽出
            for pattern, ptype in patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    if self.min_term_length <= len(match) <= self.max_term_length:
                        if match not in self.stop_words:
                            candidates[match] += 1

            # 複合語パターンでの抽出
            for pattern in compound_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    if isinstance(match, tuple):
                        # 個別の部分
                        for part in match:
                            if self.min_term_length <= len(part) <= self.max_term_length:
                                if part not in self.stop_words:
                                    candidates[part] += 1

                        # 結合形も候補に
                        combined = ''.join(match)
                        if len(combined) <= self.max_term_length:
                            candidates[combined] += 1

        # N-gram生成（隣接する名詞の結合）
        self._generate_ngrams(text, candidates)

        # 頻度フィルタリング
        return {term: freq for term, freq in candidates.items()
                if freq >= self.min_frequency}

    def _generate_ngrams(self, text: str, candidates: Counter):
        """N-gramによる複合語生成"""
        # カタカナと漢字が連続する部分を探す
        katakana_kanji = re.findall(r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+', text)
        for term in katakana_kanji:
            if self.min_term_length <= len(term) <= self.max_term_length:
                candidates[term] += 1

        # 「形容詞＋名詞」パターン
        adj_noun = re.findall(r'[一-龯]*[いきしちにひみりぎじぢびぴ]い[一-龯ァ-ヶー]+', text)
        for term in adj_noun:
            if self.min_term_length <= len(term) <= self.max_term_length:
                candidates[term] += 1

    def calculate_c_value(self, candidates: Dict[str, int], text: str) -> Dict[str, float]:
        """C-value計算（複合語の重要度）"""
        c_values = {}

        for term in candidates:
            freq = candidates[term]
            length = len(term)

            # 部分文字列として出現する回数をカウント
            nested_count = 0
            for other_term in candidates:
                if term != other_term and term in other_term:
                    nested_count += candidates[other_term]

            # C-value計算
            if nested_count == 0:
                c_value = math.log(length + 1) * freq
            else:
                c_value = math.log(length + 1) * (freq - nested_count / len([t for t in candidates if term in t and t != term]))

            c_values[term] = max(c_value, 0)

        return c_values

    def calculate_tfidf(self, candidates: Dict[str, int], all_docs: List[str]) -> Dict[str, float]:
        """TF-IDF計算"""
        tfidf_scores = {}

        # 文書頻度の計算
        doc_freq = Counter()
        for doc in all_docs:
            doc_terms = set()
            for term in candidates:
                if term in doc:
                    doc_terms.add(term)
            doc_freq.update(doc_terms)

        # TF-IDF計算
        total_terms = sum(candidates.values())
        num_docs = len(all_docs)

        for term, freq in candidates.items():
            tf = freq / total_terms
            idf = math.log(num_docs / (doc_freq.get(term, 1) + 1))
            tfidf_scores[term] = tf * idf

        return tfidf_scores

    def get_embeddings(self, terms: List[str]) -> np.ndarray:
        """埋め込みベクトル取得（キャッシュ付き）"""
        if self.use_cache:
            cache_file = self.cache_dir / f"embeddings_{hashlib.md5('_'.join(sorted(terms)).encode()).hexdigest()}.pkl"
            if cache_file.exists():
                console.print("[dim]埋め込みキャッシュを使用[/dim]")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        console.print(f"[yellow]{len(terms)}個の用語の埋め込みを計算中...[/yellow]")
        embeddings = self.embedder.encode(terms, batch_size=32, show_progress_bar=False)

        if self.use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)

        return embeddings

    def build_knn_graph(self, terms: List[str], embeddings: np.ndarray) -> nx.Graph:
        """kNNグラフ構築"""
        console.print(f"[yellow]kNNグラフを構築中 (k={self.k_neighbors})...[/yellow]")

        # kNN探索
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(terms)),
                                metric='cosine')
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # グラフ構築
        G = nx.Graph()
        G.add_nodes_from(range(len(terms)))

        for i, (dists, neighbors) in enumerate(zip(distances, indices)):
            for dist, j in zip(dists[1:], neighbors[1:]):  # 自分自身を除く
                if i != j:
                    similarity = 1 - dist
                    if similarity >= self.sim_threshold:
                        G.add_edge(i, j, weight=similarity)

        console.print(f"  [green]ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}[/green]")
        return G

    def calculate_pagerank(self, G: nx.Graph, initial_scores: Dict[int, float]) -> Dict[int, float]:
        """Personalized PageRank計算"""
        if G.number_of_nodes() == 0:
            return {}

        # Personalizationベクトル
        personalization = {i: score ** 0.7 for i, score in initial_scores.items()}
        norm = sum(personalization.values())
        if norm > 0:
            personalization = {i: v/norm for i, v in personalization.items()}

        # PageRank計算
        try:
            pagerank_scores = nx.pagerank(G, alpha=self.alpha,
                                         personalization=personalization,
                                         weight='weight')
        except:
            pagerank_scores = {i: 1.0/G.number_of_nodes() for i in G.nodes()}

        return pagerank_scores

    def extract_terms(self, pdf_path: str) -> List[Dict]:
        """PDFから専門用語を抽出"""
        console.print("\n[bold cyan]改良版専門用語抽出を開始[/bold cyan]")

        # 1. PDFからテキスト抽出
        text = self.extract_text_from_pdf(pdf_path)
        console.print(f"[dim]抽出されたテキスト: {len(text)}文字[/dim]")

        # 2. 文書分割（段落や節ごと）
        docs = re.split(r'\n\n+', text)
        docs = [d for d in docs if len(d) > 50]  # 短すぎる文書は除外

        # 3. 候補語抽出（改良版）
        console.print("\n[yellow]候補語を抽出中（複合語対応）...[/yellow]")
        candidates = self.extract_candidates_improved(text)
        console.print(f"  [green]候補数: {len(candidates)}語[/green]")

        # 4. 統計スコア計算
        console.print("\n[yellow]統計スコアを計算中...[/yellow]")
        c_values = self.calculate_c_value(candidates, text)
        tfidf_scores = self.calculate_tfidf(candidates, docs)

        # スコア統合
        terms = list(candidates.keys())
        combined_scores = {}
        for term in terms:
            c_score = c_values.get(term, 0)
            tfidf = tfidf_scores.get(term, 0)

            # 正規化
            max_c = max(c_values.values()) if c_values else 1
            max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1

            norm_c = c_score / max_c if max_c > 0 else 0
            norm_tfidf = tfidf / max_tfidf if max_tfidf > 0 else 0

            # 重み付き統合
            combined_scores[term] = 0.4 * norm_c + 0.6 * norm_tfidf

        # 5. 上位N件を選択してグラフ構築
        top_n = min(200, len(terms))
        sorted_terms = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_terms = [t[0] for t in sorted_terms[:top_n]]
        selected_indices = {term: i for i, term in enumerate(selected_terms)}

        # 6. 埋め込み計算
        console.print("\n[yellow]埋め込みベクトルを計算中...[/yellow]")
        embeddings = self.get_embeddings(selected_terms)

        # 7. kNNグラフ構築
        G = self.build_knn_graph(selected_terms, embeddings)

        # 8. PageRank計算
        console.print("\n[yellow]PageRankを計算中...[/yellow]")
        initial_scores = {i: combined_scores[term] for i, term in enumerate(selected_terms)}
        pagerank_scores = self.calculate_pagerank(G, initial_scores)

        # 9. 最終スコア統合
        final_results = []
        for i, term in enumerate(selected_terms):
            base_score = combined_scores[term]
            pr_score = pagerank_scores.get(i, 0)

            # 最終スコア（PageRank重視）
            final_score = 0.3 * base_score + 0.7 * pr_score

            final_results.append({
                'term': term,
                'score': float(final_score),
                'frequency': candidates[term],
                'c_value': float(c_values.get(term, 0)),
                'tfidf': float(tfidf_scores.get(term, 0)),
                'pagerank': float(pr_score)
            })

        # 10. ソート
        final_results.sort(key=lambda x: x['score'], reverse=True)

        return final_results


def display_results(terms: List[Dict], limit: int = 30):
    """結果を表形式で表示"""
    table = Table(title=f"抽出された専門用語（上位{limit}件）")
    table.add_column("順位", justify="right", style="cyan")
    table.add_column("用語", style="magenta")
    table.add_column("総合スコア", justify="right", style="green")
    table.add_column("頻度", justify="right", style="yellow")
    table.add_column("C-value", justify="right", style="blue")
    table.add_column("PageRank", justify="right", style="red")

    for i, term in enumerate(terms[:limit], 1):
        table.add_row(
            str(i),
            term['term'],
            f"{term['score']:.3f}",
            str(term['frequency']),
            f"{term['c_value']:.2f}",
            f"{term['pagerank']:.4f}"
        )

    console.print(table)


def analyze_domains(terms: List[Dict]):
    """ドメイン別分析（改良版）"""
    domains = {
        'エンジン・機関': ['エンジン', '機関', 'ディーゼル', 'シリンダ', 'ピストン',
                      'クランク', 'タービン', '燃焼室', 'バルブ', '噴射'],
        '燃料・化学': ['アンモニア', '燃料', '燃焼', 'NOx', 'CO2', '排気', 'ガス',
                   '水素', 'メタン', '炭化水素'],
        '船舶・海事': ['舶用', '船舶', '船級', 'IMO', '海事', '造船', 'ドック',
                   '船体', '推進', '航行'],
        '環境・規制': ['排出', '削減', 'GHG', 'ゼロエミッション', '規制', '環境',
                   '温室効果', 'カーボン', '脱炭素'],
        '技術・開発': ['開発', '実証', '試験', '評価', '技術', '研究', '設計',
                   '最適化', 'シミュレーション', '解析']
    }

    classified = {domain: [] for domain in domains}

    for term_info in terms[:50]:
        term = term_info['term']
        best_domain = None
        best_score = 0

        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in term)
            if score > best_score:
                best_score = score
                best_domain = domain

        if best_domain and best_score > 0:
            classified[best_domain].append(term)

    console.print("\n[bold cyan]ドメイン別分類（改良版）:[/bold cyan]")
    for domain, domain_terms in classified.items():
        if domain_terms:
            console.print(f"[yellow]{domain}:[/yellow]")
            console.print(f"  {', '.join(domain_terms[:15])}")


def main():
    """メイン処理"""
    # PDFファイルパス
    pdf_path = Path("input/舶用アンモニア機関の開発と実船実証.pdf")

    if not pdf_path.exists():
        console.print(f"[red]エラー: PDFファイルが見つかりません: {pdf_path}[/red]")
        return

    # 出力パス
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "improved_terms.json"

    # 抽出実行
    extractor = ImprovedTermExtractor()
    terms = extractor.extract_terms(str(pdf_path))

    # 結果表示
    console.print(f"\n[bold green]抽出完了！ 総数: {len(terms)}語[/bold green]\n")
    display_results(terms)

    # ドメイン分析
    analyze_domains(terms)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    console.print(f"\n[green]結果を保存: {output_path}[/green]")

    # 統計表示
    console.print("\n[bold cyan]統計情報:[/bold cyan]")
    console.print(f"  総候補語数: {len(terms)}")
    if terms:
        avg_score = sum(t['score'] for t in terms) / len(terms)
        avg_freq = sum(t['frequency'] for t in terms) / len(terms)
        avg_pr = sum(t['pagerank'] for t in terms) / len(terms)
        console.print(f"  平均総合スコア: {avg_score:.3f}")
        console.print(f"  平均頻度: {avg_freq:.1f}")
        console.print(f"  平均PageRank: {avg_pr:.5f}")


if __name__ == "__main__":
    main()