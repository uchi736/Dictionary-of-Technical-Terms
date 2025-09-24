# Dictionary of Technical Terms
専門用語辞書自動構築システム

## 概要
PDFドキュメントから専門用語を自動抽出し、辞書を構築するシステムです。統計的手法と機械学習を組み合わせて、高精度な専門用語抽出を実現します。

## 特徴
- 📄 PDFからの自動テキスト抽出
- 🔍 見出しやマークダウン形式の文書構造認識
- 📊 統計的手法（TF-IDF、C-value）による重要度計算
- 🧠 埋め込みベクトルとグラフベース手法（kNN + PageRank）
- ✨ 頻度が低くても重要な専門用語を見逃さない条件付きフィルタリング
- 🤖 Azure OpenAI/LLMによる専門用語検証（オプション）

## システム構成

```
dictionary_system/
├── core/
│   ├── extractors/
│   │   ├── improved_extractor.py     # メイン抽出エンジン
│   │   ├── statistical_extractor_v2.py # 統計的手法実装
│   │   └── unified_extractor.py      # 統合抽出器
│   ├── models/
│   │   └── base_extractor.py        # 基底クラス
│   └── utils/
│       └── io/
│           └── document_loader.py    # ドキュメント読み込み
├── config/
│   └── rag_config.py                # 設定管理
├── evaluation/
│   └── unknown_term_evaluator.py    # 評価ツール
├── interfaces/
│   └── extraction_interface.py      # インターフェース
└── docs/                            # ドキュメント

```

## インストール

### 必要要件
- Python 3.8以上
- 仮想環境推奨

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

### 主要な依存パッケージ
- PyMuPDF (fitz): PDF処理
- sentence-transformers: 埋め込みベクトル生成
- scikit-learn: 機械学習アルゴリズム
- networkx: グラフ処理
- numpy, pandas: データ処理
- rich: 進捗表示

## 使用方法

### 基本的な使用例

```python
from dictionary_system.core.extractors.improved_extractor import ImprovedTermExtractor

# 抽出器の初期化
extractor = ImprovedTermExtractor(
    min_frequency=2,  # 最小出現頻度
    min_term_length=2,  # 最小文字数
    max_term_length=15  # 最大文字数
)

# PDFから専門用語を抽出
terms = extractor.extract_terms("path/to/your.pdf")

# 結果の表示
for term in terms[:20]:
    print(f"{term['term']}: {term['score']:.3f}")
```

### 設定オプション

```python
extractor = ImprovedTermExtractor(
    min_frequency=1,      # 頻度閾値（1にすると見出しの用語も抽出）
    k_neighbors=10,       # kNNグラフの近傍数
    sim_threshold=0.35,   # 類似度閾値
    alpha=0.85,          # PageRankのダンピングファクター
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # 埋め込みモデル
    use_cache=True       # キャッシュ使用
)
```

## 主要機能の詳細

### 1. 見出し検出機能
マークダウン形式や番号付き見出しから専門用語を自動検出：
- `#`, `##`, `###` などのマークダウン見出し
- `1.`, `2.` などの番号付き見出し
- `第1章`, `第一節` などの日本語見出し

### 2. 条件付き頻度フィルタリング
以下の条件を満たす用語は頻度1でも候補として残す：
- 見出しに含まれる用語
- カタカナ＋漢字の組み合わせ
- 英数字を含む技術用語
- 特定のサフィックス（システム、機関、方式など）

### 3. 統計的スコアリング
- **TF-IDF**: 文書内での重要度
- **C-value**: 複合語としての重要度
- **PageRank**: 用語間の関係性による重要度

## 環境変数

`.env`ファイルで以下を設定可能：

```bash
# Azure OpenAI設定（オプション）
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=your_deployment

# LangSmith設定（オプション）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=your_project
```

## 出力例

```
抽出された専門用語（上位30件）
┏━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ 順位 ┃ 用語                  ┃ 総合スコア ┃ 頻度 ┃ C-value ┃ PageRank ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━┫
│  1  │ アンモニア燃料         │   0.892   │  57  │  68.78  │  0.0458  │
│  2  │ エンジン              │   0.831   │  42  │  52.14  │  0.0392  │
│  3  │ 燃料供給システム       │   0.756   │  23  │  41.23  │  0.0321  │
└─────┴─────────────────────┴───────────┴──────┴─────────┴──────────┘
```

## トラブルシューティング

### 文字エンコーディングエラー
Windows環境でUnicodeエラーが発生する場合：
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### メモリ不足
大きなPDFファイルの場合、バッチサイズを調整：
```python
embeddings = self.embedder.encode(terms, batch_size=16)  # デフォルト: 32
```

## ライセンス
MIT License

## 貢献
Issue や Pull Request は歓迎します。

## 作者
uchi736

## 更新履歴
- 2024.01: 見出し検出機能と条件付きフィルタリング追加
- 2024.01: 初版リリース