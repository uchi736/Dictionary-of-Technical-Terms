"""
専門用語辞書システム - メインモジュール
=====================================

このシステムは、技術文書から専門用語を抽出し、辞書を構築・管理するためのツール集です。

主な機能:
1. 用語抽出 (統計的手法、LLMベース、埋め込みベース)
2. 用語クラスタリング
3. 辞書管理 (追加、更新、削除)
4. 用語評価

使用方法:
---------
from dictionary_system import TermExtractor, DictionaryManager

# 用語抽出
extractor = TermExtractor(method="embedding")
terms = extractor.extract("your_document.txt")

# 辞書管理
manager = DictionaryManager()
manager.add_terms(terms)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from extractors.term_extractor_embeding import EmbeddingBasedExtractor
from extractors.term_extractor_with_c_value import CValueExtractor
from extractors.statistical_extractor_V2 import StatisticalExtractor
from extractors.llm_extractor_v2 import LLMExtractor

class TermExtractor:
    """統合用語抽出クラス"""

    def __init__(self, method="embedding"):
        """
        Parameters:
        -----------
        method : str
            抽出手法 ("embedding", "c_value", "statistical", "llm")
        """
        self.method = method
        self.extractor = self._initialize_extractor()

    def _initialize_extractor(self):
        """抽出器を初期化"""
        if self.method == "embedding":
            return EmbeddingBasedExtractor()
        elif self.method == "c_value":
            return CValueExtractor()
        elif self.method == "statistical":
            return StatisticalExtractor()
        elif self.method == "llm":
            return LLMExtractor()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def extract(self, text_or_file, **kwargs):
        """
        用語を抽出

        Parameters:
        -----------
        text_or_file : str
            テキストまたはファイルパス
        **kwargs : dict
            各抽出器固有のパラメータ

        Returns:
        --------
        list : 抽出された用語のリスト
        """
        return self.extractor.extract(text_or_file, **kwargs)


class DictionaryManager:
    """辞書管理クラス"""

    def __init__(self, db_path=None):
        """
        Parameters:
        -----------
        db_path : str, optional
            辞書データベースのパス
        """
        self.db_path = db_path or "dictionary.db"
        self.terms = {}

    def add_terms(self, terms):
        """用語を追加"""
        for term in terms:
            if isinstance(term, dict):
                self.terms[term.get('term', '')] = term
            else:
                self.terms[str(term)] = {'term': str(term)}

    def update_term(self, term, metadata):
        """用語のメタデータを更新"""
        if term in self.terms:
            self.terms[term].update(metadata)

    def remove_term(self, term):
        """用語を削除"""
        if term in self.terms:
            del self.terms[term]

    def get_all_terms(self):
        """全用語を取得"""
        return list(self.terms.values())

    def save(self, filepath=None):
        """辞書を保存"""
        import json
        filepath = filepath or self.db_path
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.terms, f, ensure_ascii=False, indent=2)

    def load(self, filepath=None):
        """辞書を読み込み"""
        import json
        filepath = filepath or self.db_path
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.terms = json.load(f)


if __name__ == "__main__":
    print("専門用語辞書システム")
    print("=" * 50)
    print("利用可能な抽出手法:")
    print("- embedding: 埋め込みベース")
    print("- c_value: C-value統計手法")
    print("- statistical: 統計的手法")
    print("- llm: LLMベース")