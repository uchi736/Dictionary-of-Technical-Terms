"""
Unified Term Extractor - 統合された用語抽出システム
すべての抽出手法を統一インターフェースで提供
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from dataclasses import asdict
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.models import Term, BaseExtractor


class UnifiedTermExtractor:
    """統合用語抽出器 - すべての抽出手法を統一的に扱う"""

    def __init__(self, method: str = "statistical", **kwargs):
        """
        初期化

        Args:
            method: 使用する抽出手法 ('statistical', 'llm', 'clustering', 'c_value', 'ensemble')
            **kwargs: 各抽出器に渡すパラメータ
        """
        self.method = method
        self.kwargs = kwargs
        self.extractors = {}
        self._initialize_extractors()

    def _initialize_extractors(self):
        """利用可能な抽出器を動的に初期化"""
        # Try to import available extractors
        available_extractors = {}

        # Statistical Extractor V2
        try:
            from extractors.statistical_extractor_V2 import StatisticalExtractorV2
            available_extractors['statistical'] = StatisticalExtractorV2
        except ImportError:
            pass

        # LLM Extractor
        try:
            from extractors.llm_extractor_v2 import LLMExtractorV2
            available_extractors['llm'] = LLMExtractorV2
        except ImportError:
            pass

        # Clustering Analyzer
        try:
            from extractors.term_clustering_analyzer import TermClusteringAnalyzer
            available_extractors['clustering'] = TermClusteringAnalyzer
        except ImportError:
            pass

        # C-Value Extractor
        try:
            from extractors.term_extractor_with_c_value import CValueExtractor
            available_extractors['c_value'] = CValueExtractor
        except ImportError:
            pass

        if self.method == 'ensemble':
            # アンサンブル手法の場合、すべての利用可能な抽出器を初期化
            for name, cls in available_extractors.items():
                try:
                    self.extractors[name] = cls(**self.kwargs)
                except Exception as e:
                    print(f"Warning: Could not initialize {name}: {e}")
        elif self.method in available_extractors:
            self.extractor = available_extractors[self.method](**self.kwargs)
        else:
            # デフォルトの基本抽出器を使用
            print(f"Warning: Method '{self.method}' not available. Using basic extractor.")
            self.extractor = BasicTermExtractor(**self.kwargs)

    def extract(
        self,
        text_or_file: Union[str, Path],
        output_format: str = "json",
        confidence_threshold: float = 0.5,
        max_terms: Optional[int] = None,
        **kwargs
    ) -> Union[List[Term], Dict[str, Any], str]:
        """
        用語抽出を実行

        Args:
            text_or_file: 抽出対象のテキストまたはファイルパス
            output_format: 出力形式 ('json', 'list', 'dict')
            confidence_threshold: 信頼度の閾値
            max_terms: 最大抽出用語数
            **kwargs: 追加パラメータ

        Returns:
            抽出された用語リストまたは指定形式のデータ
        """
        # ファイルかテキストかを判定
        if isinstance(text_or_file, Path) or (
            isinstance(text_or_file, str) and
            os.path.exists(text_or_file) and
            os.path.isfile(text_or_file)
        ):
            with open(text_or_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_or_file

        # 抽出実行
        if self.method == 'ensemble':
            terms = self._extract_ensemble(text, **kwargs)
        else:
            # 各抽出器のextractメソッドを呼び出し
            if hasattr(self.extractor, 'extract'):
                result = self.extractor.extract(text, **kwargs)
                # 結果をTermオブジェクトのリストに変換
                terms = self._convert_to_terms(result)
            else:
                terms = []

        # フィルタリング
        if confidence_threshold > 0:
            terms = [t for t in terms if t.score >= confidence_threshold]

        if max_terms is not None:
            terms = terms[:max_terms]

        # 出力形式変換
        return self._format_output(terms, output_format)

    def _convert_to_terms(self, result: Any) -> List[Term]:
        """抽出結果をTermオブジェクトのリストに変換"""
        terms = []

        if isinstance(result, list):
            for item in result:
                if isinstance(item, Term):
                    terms.append(item)
                elif isinstance(item, dict):
                    # 辞書からTermオブジェクトを作成
                    term_text = item.get('term', item.get('word', ''))
                    score = item.get('score', item.get('weight', 0.0))
                    if term_text:
                        terms.append(Term(
                            term=term_text,
                            score=float(score),
                            category=item.get('category'),
                            metadata=item.get('metadata', {})
                        ))
                elif isinstance(item, str):
                    # 文字列の場合はスコア1.0で作成
                    terms.append(Term(term=item, score=1.0))
                elif isinstance(item, tuple) and len(item) >= 2:
                    # タプルの場合 (term, score)
                    terms.append(Term(term=str(item[0]), score=float(item[1])))

        return terms

    def _extract_ensemble(self, text: str, **kwargs) -> List[Term]:
        """アンサンブル抽出を実行"""
        all_terms = {}

        # 各抽出器で抽出
        for name, extractor in self.extractors.items():
            try:
                if hasattr(extractor, 'extract'):
                    result = extractor.extract(text, **kwargs)
                    terms = self._convert_to_terms(result)

                    for term in terms:
                        if term.term not in all_terms:
                            all_terms[term.term] = {
                                'term': term.term,
                                'scores': {},
                                'categories': set(),
                                'metadata': {}
                            }
                        all_terms[term.term]['scores'][name] = term.score
                        if term.category:
                            all_terms[term.term]['categories'].add(term.category)
                        if term.metadata:
                            all_terms[term.term]['metadata'].update(term.metadata)
            except Exception as e:
                print(f"Error in {name} extractor: {e}")
                continue

        # スコア統合とTerm生成
        final_terms = []
        for term_text, data in all_terms.items():
            # 平均スコア計算
            if data['scores']:
                avg_score = sum(data['scores'].values()) / len(data['scores'])
            else:
                avg_score = 0.0

            # 最も頻出するカテゴリを選択
            category = max(data['categories']) if data['categories'] else None

            final_terms.append(Term(
                term=term_text,
                score=avg_score,
                category=category,
                metadata={
                    **data['metadata'],
                    'extraction_methods': list(data['scores'].keys()),
                    'method_scores': data['scores']
                }
            ))

        return sorted(final_terms, key=lambda x: x.score, reverse=True)

    def _format_output(self, terms: List[Term], format: str) -> Union[List[Term], Dict, str]:
        """出力形式を変換"""
        if format == "list":
            return terms
        elif format == "dict":
            return {
                'terms': [t.to_dict() for t in terms],
                'count': len(terms),
                'method': self.method
            }
        elif format == "json":
            return json.dumps(
                [t.to_dict() for t in terms],
                ensure_ascii=False,
                indent=2
            )
        else:
            return terms

    def save_results(
        self,
        terms: List[Term],
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """結果をファイルに保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [t.to_dict() for t in terms],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        elif format == "csv":
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['term', 'score', 'category'])
                writer.writeheader()
                for term in terms:
                    writer.writerow({
                        'term': term.term,
                        'score': term.score,
                        'category': term.category
                    })
        else:
            raise ValueError(f"Unsupported format: {format}")


class BasicTermExtractor(BaseExtractor):
    """基本的な用語抽出器（フォールバック用）"""

    def extract(self, text: str, **kwargs) -> List[Term]:
        """
        簡単な統計的手法で用語を抽出

        Args:
            text: 入力テキスト

        Returns:
            抽出された用語リスト
        """
        import re
        from collections import Counter

        # 簡単なトークン化
        words = re.findall(r'\b[a-zA-Z]+\b|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+', text.lower())

        # 単語の頻度計算
        word_freq = Counter(words)

        # Termオブジェクトに変換
        terms = []
        max_freq = max(word_freq.values()) if word_freq else 1

        for word, freq in word_freq.most_common(100):
            if len(word) > 2:  # 2文字以上の単語のみ
                terms.append(Term(
                    term=word,
                    score=freq / max_freq
                ))

        return terms