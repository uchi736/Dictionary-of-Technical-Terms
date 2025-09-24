# 専門用語辞書システム

技術文書から専門用語を自動抽出し、辞書を構築・管理するシステムです。

## 機能

### 1. 用語抽出手法
- **埋め込みベース** (`term_extractor_embeding.py`)
  - 文脈を考慮した高精度な抽出
  - OpenAI Embeddingsを使用

- **C-value法** (`term_extractor_with_c_value.py`)
  - 統計的手法による複合語抽出
  - 形態素解析ベース

- **統計的手法** (`statistical_extractor_V2.py`)
  - TF-IDF、出現頻度ベース
  - 高速処理

- **LLMベース** (`llm_extractor_v2.py`)
  - GPT-4等による文脈理解
  - 高精度だが処理コスト大

### 2. 用語管理
- **辞書更新** (`term_dictionary_updater.py`)
- **DB管理** (`import_terms_to_db.py`, `clear_jargon_db.py`)
- **クラスタリング** (`term_clustering_analyzer.py`)

## ディレクトリ構造
```
dictionary_system/
├── extractors/        # 用語抽出モジュール
├── evaluation/        # 評価ツール
├── utils/            # ユーティリティ
├── data/             # データファイル
├── docs/             # ドキュメント
├── config/           # 設定ファイル
└── main.py           # メインモジュール
```

## 使用方法

### 基本的な使用例
```python
from dictionary_system.main import TermExtractor, DictionaryManager

# 用語抽出
extractor = TermExtractor(method="embedding")
terms = extractor.extract("path/to/document.txt")

# 辞書管理
manager = DictionaryManager()
manager.add_terms(terms)
manager.save("my_dictionary.json")
```

### 各抽出手法の使い分け
| 手法 | 精度 | 速度 | 用途 |
|------|------|------|------|
| 埋め込み | 高 | 中 | 技術文書、専門分野 |
| C-value | 中 | 高 | 複合語が多い文書 |
| 統計的 | 中 | 高 | 大量文書の高速処理 |
| LLM | 最高 | 低 | 少量の重要文書 |

## 必要なライブラリ
```
openai
langchain
scikit-learn
pandas
numpy
mecab-python3
```

## 環境変数
`.env`ファイルに以下を設定:
```
OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_KEY=your_azure_key  # Azure使用時
```