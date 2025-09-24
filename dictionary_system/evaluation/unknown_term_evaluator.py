#!/usr/bin/env python3
"""
LLM未知語評価システム
専門用語の定義をコンテキストなしで生成し、
元の定義とのベクトル類似度を比較して真の未知語を特定
"""

import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI
from rich.console import Console
from rich.table import Table
from rich.progress import track

load_dotenv()
console = Console()


class UnknownTermEvaluator:
    """LLM未知語評価器"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        llm_model: str = "gemini-2.0-flash-exp",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        初期化
        
        Args:
            similarity_threshold: 未知語判定の類似度閾値（これ以下なら未知語）
            llm_model: 定義生成に使用するLLMモデル
            embedding_model: ベクトル化に使用する埋め込みモデル
        """
        self.similarity_threshold = similarity_threshold
        
        # LLM設定（定義生成用）
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.1,
            max_tokens=200
        )
        self.llm_model_name = llm_model
        
        # Azure OpenAI設定（埋め込み用）
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
            embedding_model
        )
        
        # プロンプトテンプレートは動的に生成
        
    def generate_context_free_definition(self, term: str, field: str = None, retry_count: int = 3) -> str:
        """
        コンテキストなしで用語の定義を生成（改良版）
        
        Args:
            term: 定義を生成する用語
            field: 分野情報
            retry_count: リトライ回数
            
        Returns:
            生成された定義
        """
        # 分野別プロンプト調整
        field_hint = f"（{field}分野）" if field else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門用語辞典の編集者です。
一般的な事前知識の範囲のみで定義してください。
確信が持てない場合は「定義不明」と答えてください。"""),
            ("user", """次の用語{field_hint}を1-2文で定義してください。
推測や曖昧な説明は避け、わからない場合は「定義不明」と答えてください。

用語: {term}""")
        ])
        
        for attempt in range(retry_count):
            try:
                chain = prompt | self.llm
                response = chain.invoke({"term": term, "field_hint": field_hint})
                return response.content.strip()
            except Exception as e:
                if "429" in str(e) and attempt < retry_count - 1:
                    # Rate limit hit, wait before retry
                    wait_time = (attempt + 1) * 10  # 10, 20, 30 seconds
                    console.print(f"[yellow]Rate limit hit for {term}, waiting {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
                else:
                    console.print(f"[red]Error generating definition for {term}: {e}[/red]")
                    return ""
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        テキストの埋め込みベクトルを取得
        
        Args:
            text: ベクトル化するテキスト
            
        Returns:
            埋め込みベクトル
        """
        try:
            response = self.azure_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            console.print(f"[red]Error getting embedding: {e}[/red]")
            return np.array([])
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        コサイン類似度を計算
        
        Args:
            vec1: ベクトル1
            vec2: ベクトル2
            
        Returns:
            コサイン類似度（0-1）
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def evaluate_term(self, term_data: Dict) -> Dict:
        """
        単一の用語を評価
        
        Args:
            term_data: 用語データ（term, definition含む）
            
        Returns:
            評価結果
        """
        term = term_data.get("term", "")
        original_definition = term_data.get("definition", "")
        field = term_data.get("metadata", {}).get("field", None)
        
        # コンテキストなし定義を生成（分野情報付き）
        context_free_definition = self.generate_context_free_definition(term, field)
        
        if not context_free_definition:
            return {
                "term": term,
                "error": "Failed to generate context-free definition",
                "is_unknown": False
            }
        
        # 「定義不明」判定を追加
        if any(keyword in context_free_definition for keyword in 
               ["定義不明", "わかりません", "不明", "特定できません"]):
            return {
                "term": term,
                "original_definition": original_definition,
                "context_free_definition": context_free_definition,
                "similarity": 0.0,
                "is_unknown": True,
                "explicitly_unknown": True,  # 明示的に未知と判定
                "metadata": term_data.get("metadata", {})
            }
        
        # 両方の定義をベクトル化
        original_embedding = self.get_embedding(original_definition)
        context_free_embedding = self.get_embedding(context_free_definition)
        
        # 類似度計算
        similarity = self.calculate_cosine_similarity(
            original_embedding,
            context_free_embedding
        )
        
        # 未知語判定
        is_unknown = similarity < self.similarity_threshold
        
        return {
            "term": term,
            "original_definition": original_definition,
            "context_free_definition": context_free_definition,
            "similarity": float(similarity),
            "is_unknown": is_unknown,
            "explicitly_unknown": False,
            "metadata": term_data.get("metadata", {})
        }
    
    def evaluate_terms_from_file(self, input_file: str) -> Dict:
        """
        ファイルから用語を読み込んで評価
        
        Args:
            input_file: llm_terms.jsonファイルパス
            
        Returns:
            評価結果
        """
        # ファイル読み込み
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_terms = []
        
        # 各ファイルの結果から用語を収集
        for result in data.get("results", []):
            terms = result.get("terms", [])
            all_terms.extend(terms)
        
        console.print(f"[cyan]Evaluating {len(all_terms)} terms...[/cyan]")
        
        # 評価実行（Rate limit対策のため少し間隔を開ける）
        evaluated_terms = []
        for i, term_data in enumerate(track(all_terms, description="Evaluating terms")):
            evaluation = self.evaluate_term(term_data)
            evaluated_terms.append(evaluation)
            
            # Rate limit対策：5件ごとに少し待つ
            if (i + 1) % 5 == 0:
                time.sleep(2)
        
        # 結果を分類
        true_unknown_terms = [t for t in evaluated_terms if t.get("is_unknown")]
        explicitly_unknown = [t for t in true_unknown_terms if t.get("explicitly_unknown")]
        similarity_unknown = [t for t in true_unknown_terms if not t.get("explicitly_unknown")]
        known_terms = [t for t in evaluated_terms if not t.get("is_unknown") and "error" not in t]
        error_terms = [t for t in evaluated_terms if "error" in t]
        
        return {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "total_terms": len(all_terms),
                "similarity_threshold": self.similarity_threshold,
                "llm_model": self.llm_model_name,
                "embedding_model": self.embedding_deployment
            },
            "summary": {
                "true_unknown_count": len(true_unknown_terms),
                "explicitly_unknown_count": len(explicitly_unknown),
                "similarity_unknown_count": len(similarity_unknown),
                "known_count": len(known_terms),
                "error_count": len(error_terms)
            },
            "true_unknown_terms": true_unknown_terms,
            "known_terms": known_terms,
            "error_terms": error_terms
        }
    
    def display_results(self, results: Dict):
        """
        評価結果を表示
        
        Args:
            results: 評価結果
        """
        console.print("\n[bold green]== Evaluation Results ==[/bold green]")
        
        # サマリー表示
        summary = results["summary"]
        console.print(f"Total terms evaluated: {results['metadata']['total_terms']}")
        console.print(f"Similarity threshold: {results['metadata']['similarity_threshold']}")
        console.print(f"True unknown terms: [red]{summary['true_unknown_count']}[/red]")
        console.print(f"  - Explicitly unknown (定義不明): [bold red]{summary['explicitly_unknown_count']}[/bold red]")
        console.print(f"  - Low similarity: [red]{summary['similarity_unknown_count']}[/red]")
        console.print(f"Known terms: [green]{summary['known_count']}[/green]")
        console.print(f"Errors: [yellow]{summary['error_count']}[/yellow]")
        
        # 真の未知語テーブル
        if results["true_unknown_terms"]:
            console.print("\n[bold red]True Unknown Terms (Low Similarity)[/bold red]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Term", style="cyan", width=30)
            table.add_column("Similarity", justify="right", width=10)
            table.add_column("Original Definition", width=40)
            table.add_column("Context-Free Definition", width=40)
            
            for term in results["true_unknown_terms"][:10]:  # 最初の10件を表示
                table.add_row(
                    term["term"],
                    f"{term['similarity']:.3f}",
                    term["original_definition"][:60] + "...",
                    term["context_free_definition"][:60] + "..."
                )
            
            console.print(table)
        
        # 既知語の例
        if results["known_terms"]:
            console.print("\n[bold green]Sample Known Terms (High Similarity)[/bold green]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Term", style="cyan", width=30)
            table.add_column("Similarity", justify="right", width=10)
            
            for term in results["known_terms"][:5]:  # 最初の5件を表示
                table.add_row(
                    term["term"],
                    f"{term['similarity']:.3f}"
                )
            
            console.print(table)
    
    def save_results(self, results: Dict, output_file: str):
        """
        評価結果を保存
        
        Args:
            results: 評価結果
            output_file: 出力ファイルパス
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]Results saved to {output_path}[/green]")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate LLM unknown terms using vector similarity"
    )
    parser.add_argument(
        "input",
        type=str,
        default="output/llm_terms.json",
        nargs="?",
        help="Input JSON file with extracted terms"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/unknown_term_evaluation.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for unknown term detection (default: 0.7)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="LLM model for definition generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model for vectorization"
    )
    
    args = parser.parse_args()
    
    # 評価器を初期化
    evaluator = UnknownTermEvaluator(
        similarity_threshold=args.threshold,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model
    )
    
    # 評価実行
    results = evaluator.evaluate_terms_from_file(args.input)
    
    # 結果表示
    evaluator.display_results(results)
    
    # 結果保存
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()