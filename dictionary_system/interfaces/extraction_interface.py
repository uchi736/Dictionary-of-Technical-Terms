"""
Extraction Interface - 用語抽出システムのメインインターフェース
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors.unified_extractor import UnifiedTermExtractor
from core.models import Term
from core.utils.io.document_loader import DocumentLoader


class ExtractionInterface:
    """用語抽出のメインインターフェース"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.document_loader = DocumentLoader()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_extraction(
        self,
        input_path: str,
        method: str = "statistical",
        output_path: Optional[str] = None,
        output_format: str = "json",
        confidence_threshold: float = 0.5,
        max_terms: Optional[int] = None,
        **kwargs
    ) -> List[Term]:
        """
        用語抽出を実行

        Args:
            input_path: 入力ファイルのパス
            method: 抽出手法 ('statistical', 'llm', 'clustering', 'c_value', 'ensemble')
            output_path: 出力ファイルのパス
            output_format: 出力形式 ('json', 'csv')
            confidence_threshold: 信頼度の閾値
            max_terms: 最大抽出用語数
            **kwargs: 追加パラメータ

        Returns:
            抽出された用語リスト
        """
        print(f"Processing: {input_path}")
        print(f"Method: {method}")

        # ドキュメントを読み込み
        try:
            text = self.document_loader.load(input_path)
        except Exception as e:
            print(f"Error loading document: {e}")
            return []

        # 抽出器の初期化
        extractor = UnifiedTermExtractor(method=method, **kwargs)

        # 抽出実行
        terms = extractor.extract(
            text_or_file=text,
            output_format="list",
            confidence_threshold=confidence_threshold,
            max_terms=max_terms,
            **kwargs
        )

        # 結果保存
        if output_path:
            extractor.save_results(terms, output_path, format=output_format)
            print(f"Results saved to: {output_path}")

        return terms

    def batch_extraction(
        self,
        input_dir: str,
        output_dir: str,
        method: str = "statistical",
        file_pattern: str = "*",
        recursive: bool = False,
        **kwargs
    ) -> Dict[str, List[Term]]:
        """
        バッチ処理で複数ファイルから用語抽出

        Args:
            input_dir: 入力ディレクトリのパス
            output_dir: 出力ディレクトリのパス
            method: 抽出手法
            file_pattern: ファイルパターン
            recursive: サブディレクトリも処理するか
            **kwargs: 追加パラメータ

        Returns:
            ファイル名と抽出結果の辞書
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # ファイルを取得
        documents = self.document_loader.load_directory(
            input_dir,
            pattern=file_pattern,
            recursive=recursive
        )

        print(f"Found {len(documents)} files to process")

        # 各ファイルを処理
        for file_path, text in documents:
            file_name = Path(file_path).name
            print(f"\nProcessing: {file_name}")

            output_path = output_dir / f"{Path(file_path).stem}_extracted.json"

            # 抽出器の初期化
            extractor = UnifiedTermExtractor(method=method, **kwargs)

            # 抽出実行
            terms = extractor.extract(
                text_or_file=text,
                output_format="list",
                **kwargs
            )

            # 結果保存
            extractor.save_results(terms, output_path, format="json")

            results[file_name] = terms
            print(f"  Extracted {len(terms)} terms")

        return results

    def compare_methods(
        self,
        input_path: str,
        output_dir: str,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, List[Term]]:
        """
        複数の抽出手法を比較

        Args:
            input_path: 入力ファイルのパス
            output_dir: 出力ディレクトリのパス
            methods: 比較する手法のリスト
            **kwargs: 追加パラメータ

        Returns:
            手法名と抽出結果の辞書
        """
        if methods is None:
            methods = ["statistical", "c_value", "ensemble"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        print(f"Comparing methods: {', '.join(methods)}")

        for method in methods:
            print(f"\n{'='*50}")
            print(f"Method: {method}")
            print('='*50)

            output_path = output_dir / f"{method}_results.json"

            terms = self.run_extraction(
                input_path=input_path,
                method=method,
                output_path=str(output_path),
                **kwargs
            )

            results[method] = terms
            print(f"Extracted {len(terms)} terms")

        # 比較結果を保存
        comparison_path = output_dir / "comparison_summary.json"
        comparison_data = {}

        for method, terms in results.items():
            comparison_data[method] = {
                "count": len(terms),
                "top_10": [
                    {"term": t.term, "score": t.score}
                    for t in terms[:10]
                ]
            }

        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)

        print(f"\nComparison saved to: {comparison_path}")

        return results


def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(
        description="Technical Term Extraction System - 技術用語抽出システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 単一ファイル処理
  python extraction_interface.py input.txt -m statistical -o output.json

  # バッチ処理
  python extraction_interface.py input_dir/ -o output_dir/ --batch

  # 手法比較
  python extraction_interface.py input.txt -o comparison/ --compare

  # カスタム設定
  python extraction_interface.py input.txt -m ensemble -t 0.7 -n 50
        """
    )

    parser.add_argument(
        "input",
        help="Input file or directory path"
    )

    parser.add_argument(
        "-m", "--method",
        default="statistical",
        choices=["statistical", "llm", "clustering", "c_value", "ensemble"],
        help="Extraction method to use (default: statistical)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file or directory path"
    )

    parser.add_argument(
        "-f", "--format",
        default="json",
        choices=["json", "csv"],
        help="Output format (default: json)"
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering terms (default: 0.5)"
    )

    parser.add_argument(
        "-n", "--max-terms",
        type=int,
        help="Maximum number of terms to extract"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple files in batch mode"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process files recursively in subdirectories"
    )

    parser.add_argument(
        "--pattern",
        default="*",
        help="File pattern for batch processing (default: *)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple extraction methods"
    )

    parser.add_argument(
        "--compare-methods",
        nargs="+",
        help="Methods to compare (default: statistical c_value ensemble)"
    )

    parser.add_argument(
        "-c", "--config",
        help="Configuration file path"
    )

    args = parser.parse_args()

    # インターフェース初期化
    interface = ExtractionInterface(
        config_path=Path(args.config) if args.config else None
    )

    # 比較モード
    if args.compare:
        if not args.output:
            print("Error: Output directory required for comparison mode")
            sys.exit(1)

        results = interface.compare_methods(
            input_path=args.input,
            output_dir=args.output,
            methods=args.compare_methods,
            confidence_threshold=args.threshold,
            max_terms=args.max_terms
        )

        print(f"\n{'='*50}")
        print("Comparison Summary:")
        print('='*50)
        for method, terms in results.items():
            print(f"{method}: {len(terms)} terms")

    # バッチ処理モード
    elif args.batch:
        if not args.output:
            print("Error: Output directory required for batch mode")
            sys.exit(1)

        results = interface.batch_extraction(
            input_dir=args.input,
            output_dir=args.output,
            method=args.method,
            file_pattern=args.pattern,
            recursive=args.recursive,
            confidence_threshold=args.threshold,
            max_terms=args.max_terms
        )

        print(f"\n{'='*50}")
        print(f"Processed {len(results)} files")
        print('='*50)
        total_terms = sum(len(terms) for terms in results.values())
        print(f"Total terms extracted: {total_terms}")

    # 単一ファイル処理モード
    else:
        terms = interface.run_extraction(
            input_path=args.input,
            method=args.method,
            output_path=args.output,
            output_format=args.format,
            confidence_threshold=args.threshold,
            max_terms=args.max_terms
        )

        print(f"\n{'='*50}")
        print(f"Extracted {len(terms)} terms")
        print('='*50)

        # 上位10件を表示
        if terms:
            print("\nTop 10 terms:")
            for i, term in enumerate(terms[:10], 1):
                print(f"{i:2}. {term.term:<30} (score: {term.score:.3f})")


if __name__ == "__main__":
    main()