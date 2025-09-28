# Dictionary of Technical Terms
専門用語辞書自動構築システム（V4: 統合候補抽出 + SemRe-Rank + RAG定義生成）

## 概要
PDFドキュメントから専門用語を自動抽出し、RAGによる定義生成で辞書を構築するシステムです。
正規表現・SudachiPy・n-gramの3手法統合による高網羅性抽出と、SemRe-Rankアルゴリズムによる高精度スコアリング、
さらにRAGによる自動定義生成を組み合わせた最新の専門用語処理システムです。

## 特徴
- 📄 PDFからの自動テキスト抽出
- 🔬 **3手法統合抽出**: 正規表現（5パターン）+ SudachiPy形態素解析 + n-gram複合語生成
- 🎯 **SemRe-Rank**: 意味的関連性 + Personalized PageRankによる高精度抽出
- 🤖 **RAG定義生成**: BM25 + ベクトル検索 + LLMによる自動定義付与
- ⚡ **LCEL対応**: LangChain Expression Languageによる宣言的パイプライン
- 🌐 **汎用性**: ドメイン非依存、統計的手法のみでフィルタリング
- 🔍 Azure OpenAI Embeddings (text-embedding-3-small) 対応
- 📊 ハイブリッド検索（BM25 + pgvector）による高精度RAG
- 🏗️ **階層的類義語抽出**: HDBSCAN + LLMによるカテゴリ命名

## システム構成

```
dictionary_system/
├── config/
│   ├── prompts.py                         # LLMプロンプト管理
│   └── rag_config.py                      # 環境設定
├── core/
│   ├── models/
│   │   └── base_extractor.py             # Term, BaseExtractor
│   ├── extractors/
│   │   ├── enhanced_term_extractor_v4.py # V4統合抽出器（推奨）
│   │   ├── semrerank_correct.py          # SemRe-Rank実装
│   │   └── statistical_extractor_v2.py   # V3実装
│   └── rag/
│       ├── bm25_index.py                  # BM25検索 + RRF
│       ├── hybrid_search.py               # ハイブリッド検索チェーン
│       ├── definition_enricher.py         # 定義付与統合
│       ├── synonym_extractor.py           # 階層的類義語抽出
│       └── extraction_pipeline.py         # LCEL パイプライン
├── ARCHITECTURE.md                        # 詳細アーキテクチャドキュメント
└── README.md                              # このファイル
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

### 1. V4統合抽出器（推奨）

```python
from dictionary_system.core.extractors.enhanced_term_extractor_v4 import EnhancedTermExtractorV4

# 抽出器の初期化
extractor = EnhancedTermExtractorV4(
    # 候補抽出パラメータ
    min_term_length=2,
    max_term_length=15,
    min_frequency=2,
    use_sudachi=True,              # SudachiPy形態素解析を使用
    use_ngram_generation=True,     # n-gram複合語生成を使用
    max_ngram=3,                   # n-gramの最大長

    # SemRe-Rankパラメータ
    relmin=0.5,                    # 最小類似度閾値
    reltop=0.15,                   # 上位選択割合
    alpha=0.85,                    # PPRダンピングファクタ

    # RAG/LLMパラメータ
    enable_definition_generation=True,    # 定義生成
    enable_definition_filtering=True,     # LLM専門用語判定
    top_n_definition=30,

    # 階層化パラメータ
    enable_synonym_hierarchy=True,        # 階層的類義語抽出
    min_cluster_size=2,
    generate_category_names=True
)

# テキストから専門用語を抽出
text = """
舶用アンモニア燃料エンジンは、次世代の環境対応技術として注目されている。
この6L28ADFエンジンは、従来のディーゼルエンジンと比較して、
GHG排出量を大幅に削減できる。国際エネルギー機関（IEA）も
この技術の重要性を指摘している。
"""

result = extractor.extract_terms(text)

# 結果の表示
print(f"抽出された専門用語: {len(result['terms'])}件")
for term in result['terms'][:10]:
    print(f"\n【{term.term}】")
    print(f"スコア: {term.score:.4f}")
    if term.definition:
        print(f"定義: {term.definition}")

# 階層構造の表示
if 'hierarchy' in result and result['hierarchy']:
    print("\n\n=== 階層的類義語グループ ===")
    for group in result['hierarchy']:
        print(f"\n◆ {group['category_name']}")
        print(f"  用語: {', '.join(group['terms'][:5])}")
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

### V4統合候補抽出（3手法）
1. **正規表現パターンマッチング（5パターン）**
   - カタカナ、漢字、英数字、混合パターン
   - 型番完全抽出（例: "6L28ADF"）

2. **SudachiPy形態素解析**
   - 連続名詞の自動連結
   - 後方suffix生成（"ABC" → "BC", "C"）

3. **n-gram複合語生成**
   - 2-gram〜max_ngramの組み合わせ
   - 見逃しを最小化（例: "舶用アンモニア", "アンモニア燃料"）

### SemRe-Rank アルゴリズム
- **基底スコア**: 0.7 × TF-IDF + 0.3 × C-value
- **シード用語選択**: エルボー法による自動境界検出
- **意味的関連性グラフ**: relmin（最小類似度）+ reltop（上位%）
- **Personalized PageRank**: シード用語から伝播
- **最終スコア**: base_score × PPR_score

### RAG定義生成
- **ハイブリッド検索**: BM25（キーワード）+ ベクトル検索（意味）
- **RRF統合**: Reciprocal Rank Fusion で検索結果を最適統合
- **LLMプロンプト**: config/prompts.py で一元管理
- **LCEL対応**: 宣言的なチェーン構築、バッチ処理高速化

### 階層的類義語抽出
- **HDBSCAN**: 密度ベースクラスタリング
- **condensed_tree**: 階層構造自動検出
- **LLMカテゴリ命名**: GPT-4oによる自動命名

### プロンプト管理
```python
from dictionary_system.config.prompts import PromptConfig

# プロンプトのカスタマイズ
config = PromptConfig()
config.definition_system = "あなたは専門用語の定義作成の専門家です..."
```

## アーキテクチャ

### データフロー（V4）
```
テキスト
  ↓
STEP 1: 候補抽出（3手法統合）
  ├─ 正規表現（5パターン）
  ├─ SudachiPy形態素解析 + suffix
  └─ n-gram複合語生成
  ↓
STEP 2-3: 統計的スコアリング
  ├─ TF-IDF計算
  └─ C-value計算
  ↓
STEP 4-6: SemRe-Rank
  ├─ シード選択（エルボー法）
  ├─ 意味的関連性グラフ（relmin/reltop）
  └─ Personalized PageRank
  ↓
STEP 7: RAG定義生成（オプション）
  ├─ BM25検索
  ├─ ベクトル検索（pgvector）
  ├─ RRF統合
  └─ LLM定義生成
  ↓
STEP 8: LLM専門用語判定（オプション）
  └─ 用語+定義をLLMで判定
  ↓
STEP 9: 階層的類義語抽出（オプション）
  ├─ HDBSCAN クラスタリング
  └─ LLM カテゴリ命名
  ↓
専門用語辞書 + 階層構造
```

詳細は [ARCHITECTURE.md](ARCHITECTURE.md) を参照してください。

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
- **SudachiPy**: 日本語形態素解析
- **sentence-transformers**: 埋め込みベクトル
- **networkx**: グラフ処理（PageRank）
- **scikit-learn**: TF-IDF、HDBSCAN
- **numpy, pandas**: データ処理

## ライセンス
MIT License

## 参考文献
- Zhang et al. (2017). "SemRe-Rank: Improving Automatic Term Extraction By Incorporating Semantic Relatedness With Personalised PageRank"

## 作者
uchi736

## 更新履歴
- 2025.01: **V4リリース** - 3手法統合候補抽出（正規表現5パターン + SudachiPy + n-gram）、階層的類義語抽出、汎用性向上
- 2025.01: SemRe-Rank + RAG定義生成 + LCEL対応（メジャーアップデート）
- 2024.01: 見出し検出機能と条件付きフィルタリング追加
- 2024.01: 初版リリース