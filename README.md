# Dictionary of Technical Terms
専門用語辞書自動構築システム（SemRe-Rank + RAG定義生成）

## 概要
PDFドキュメントから専門用語を自動抽出し、RAGによる定義生成で辞書を構築するシステムです。
SemRe-Rankアルゴリズムと最新のLangChain LCEL技術を組み合わせた高精度な専門用語処理システムです。

## 特徴
- 📄 PDFからの自動テキスト抽出
- 🔬 **SemRe-Rank**: 意味的関連性 + Personalized PageRankによる高精度抽出
- 🤖 **RAG定義生成**: BM25 + ベクトル検索 + LLMによる自動定義付与
- ⚡ **LCEL対応**: LangChain Expression Languageによる宣言的パイプライン
- 🎯 エルボー法によるシード用語自動選択
- 🔍 Azure OpenAI Embeddings (text-embedding-3-small) 対応
- 📊 ハイブリッド検索（BM25 + pgvector）による高精度RAG

## システム構成

```
dictionary_system/
├── config/
│   ├── prompts.py                  # LLMプロンプト管理
│   └── rag_config.py               # 環境設定
├── core/
│   ├── models/
│   │   └── base_extractor.py      # Term, BaseExtractor
│   ├── extractors/
│   │   ├── semrerank_correct.py   # SemRe-Rank実装
│   │   └── statistical_extractor_v2.py
│   └── rag/
│       ├── bm25_index.py           # BM25検索 + RRF
│       ├── hybrid_search.py        # ハイブリッド検索チェーン
│       ├── definition_enricher.py  # 定義付与統合
│       └── extraction_pipeline.py  # LCEL パイプライン
└── docs/                           # ドキュメント
```

## インストール

### 必要要件
- Python 3.8以上
- PostgreSQL + pgvector（RAG機能使用時）
- Azure OpenAI アカウント（推奨）

### セットアップ
```bash
# リポジトリのクローン
git clone https://github.com/uchi736/Dictionary-of-Technical-Terms.git
cd Dictionary-of-Technical-Terms

# 仮想環境の作成と有効化
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 環境変数設定
`.env`ファイルを作成：

```bash
# Azure OpenAI設定
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# PostgreSQL + pgvector設定（RAG機能使用時）
DB_HOST=your_host
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password

# LangSmith設定（オプション）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=term-extraction
```

## 使用方法

### 1. 基本的な専門用語抽出（SemRe-Rank）

```python
from dictionary_system.core.extractors.semrerank_correct import SemReRankExtractor

# 抽出器の初期化
extractor = SemReRankExtractor(
    base_ate_method="tfidf",
    use_azure_embeddings=True,
    auto_select_seeds=True,
    seed_z=50,
    use_elbow_detection=True,
    min_seed_count=5,
    max_seed_ratio=0.7
)

# テキストから専門用語を抽出
text = """
アンモニア燃料エンジンは、次世代の環境対応技術として注目されている。
このアンモニア燃料エンジンは、従来のディーゼルエンジンと比較して、
CO2排出量を大幅に削減できる。
"""

terms = extractor.extract(text)

# 結果の表示
for term in terms[:10]:
    print(f"{term.term}: {term.score:.4f}")
```

### 2. 定義自動生成（RAG）

```python
from dictionary_system.core.rag import enrich_terms_with_definitions

# 抽出した用語に定義を付与
enriched_terms = enrich_terms_with_definitions(
    terms=terms,
    text=text,
    top_n=5,
    verbose=True
)

# 結果の表示
for term in enriched_terms[:5]:
    print(f"\n【{term.term}】")
    print(f"スコア: {term.score:.4f}")
    print(f"定義: {term.definition}")
```

### 3. エンドツーエンドパイプライン（LCEL）

```python
from dictionary_system.core.rag import create_extraction_pipeline

# パイプラインの作成
pipeline = create_extraction_pipeline(
    enable_definitions=True,
    top_n_terms=10
)

# テキストから用語抽出+定義生成を一気に実行
result = pipeline.invoke(text)

print(f"抽出された専門用語: {result['count']}件")
for term in result['top_terms']:
    print(f"\n【{term.term}】")
    print(f"定義: {term.definition}")
```

### 4. バッチ処理

```python
# 複数テキストの並列処理
texts = [text1, text2, text3]
results = pipeline.batch(texts)

# 非同期処理
import asyncio
results = await pipeline.abatch(texts)
```

## 主要機能の詳細

### SemRe-Rank アルゴリズム
- **シード用語選択**: エルボー法による自動境界検出
- **意味的関連性**: Azure埋め込み or SentenceTransformer
- **Personalized PageRank**: シード用語を重視したグラフスコアリング
- **文書レベル処理**: 文単位でグラフ構築→集約

### RAG定義生成
- **ハイブリッド検索**: BM25（キーワード）+ ベクトル検索（意味）
- **RRF統合**: Reciprocal Rank Fusion で検索結果を最適統合
- **LLMプロンプト**: config/prompts.py で一元管理
- **LCEL対応**: 宣言的なチェーン構築、バッチ処理高速化

### プロンプト管理
```python
from dictionary_system.config.prompts import PromptConfig

# プロンプトのカスタマイズ
config = PromptConfig()
config.definition_system = "あなたは専門用語の定義作成の専門家です..."
```

## アーキテクチャ

### データフロー
```
テキスト
  ↓
前処理
  ↓
SemRe-Rank 抽出
  ├─ 候補抽出（正規表現）
  ├─ TF-IDF スコアリング
  ├─ シード選択（エルボー法）
  └─ Personalized PageRank
  ↓
用語リスト（Term[]）
  ↓
RAG 定義生成
  ├─ BM25 検索
  ├─ ベクトル検索（pgvector）
  ├─ RRF 統合
  └─ LLM 定義生成
  ↓
定義付き用語辞書
```

### 依存関係
```
レイヤー0: base_extractor, rag_config, prompts
レイヤー1: bm25_index
レイヤー2: semrerank_correct
レイヤー3: hybrid_search
レイヤー4: definition_enricher
レイヤー5: extraction_pipeline
```

## パフォーマンス

- **バッチ処理**: LCEL の .batch() で並列化
- **非同期処理**: .abatch() で高速化
- **キャッシュ**: 埋め込みベクトルのキャッシュ対応
- **ストリーミング**: .stream() で部分結果取得

## トラブルシューティング

### pgvector 接続エラー
PostgreSQL + pgvector 拡張が必要です：
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### MeCab エラー
BM25 で MeCab を使用する場合：
```bash
# Windows
pip install mecab-python3
# Windowsの場合は別途MeCabのインストールが必要

# Linux/Mac
pip install mecab-python3
```

MeCab がなくても動作します（文字単位のトークン化にフォールバック）。

## 主要依存パッケージ
- **LangChain**: LCEL パイプライン、RAG統合
- **langchain-openai**: Azure OpenAI 統合
- **langchain-postgres**: pgvector 連携
- **sentence-transformers**: 埋め込みベクトル
- **networkx**: グラフ処理（PageRank）
- **scikit-learn**: TF-IDF
- **numpy, pandas**: データ処理

## ライセンス
MIT License

## 参考文献
- Zhang et al. (2017). "SemRe-Rank: Improving Automatic Term Extraction By Incorporating Semantic Relatedness With Personalised PageRank"

## 作者
uchi736

## 更新履歴
- 2025.01: SemRe-Rank + RAG定義生成 + LCEL対応（メジャーアップデート）
- 2024.01: 見出し検出機能と条件付きフィルタリング追加
- 2024.01: 初版リリース