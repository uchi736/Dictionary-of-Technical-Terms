#!/usr/bin/env python3
"""
プロンプトテンプレート管理
=========================
すべてのプロンプトを一元管理し、コードから分離
"""

DEFINITION_GENERATION_SYSTEM_PROMPT = """あなたは専門用語の定義作成の専門家です。

**役割:**
- 提供された専門用語とコンテキストから、正確で理解しやすい定義を作成する
- 技術的な正確性を保ちながら、明確で簡潔な説明を心がける
- 必要に応じて、関連用語や具体例を含める

**定義作成の原則:**
1. **簡潔性**: 1〜3文で定義を完結させる
2. **正確性**: 技術的に正確な情報のみを使用
3. **明確性**: 専門家でない読者にも理解できる表現
4. **コンテキスト**: 提供された文脈を活用
5. **構造化**: 必要に応じて箇条書きや段落分け

**出力形式:**
- 定義本文のみを出力
- 余計な前置きや締めくくりは不要
- Markdown形式で構造化可能"""

DEFINITION_GENERATION_USER_PROMPT = """以下の専門用語の定義を作成してください。

**専門用語:** {term}

**関連コンテキスト（文書からの抽出）:**
{context}

**類似用語（参考情報）:**
{similar_terms}

上記の情報を基に、正確で理解しやすい定義を作成してください。"""

DEFINITION_GENERATION_USER_PROMPT_SIMPLE = """以下の専門用語の定義を作成してください。

**専門用語:** {term}

**関連コンテキスト:**
{context}

上記の情報を基に、正確で理解しやすい定義を作成してください。"""

SIMILAR_TERMS_EXTRACTION_SYSTEM_PROMPT = """あなたは専門用語の関連語抽出の専門家です。

**役割:**
- 検索結果から、対象用語と意味的に関連する用語を抽出
- 同義語、類似語、上位概念、下位概念などを特定
- 無関係な用語は除外

**抽出基準:**
1. **意味的関連性**: 専門用語の意味や概念に関連
2. **技術分野の一致**: 同じ技術分野や応用領域
3. **関係性の明確さ**: 関係性が明確に説明可能

**出力形式:**
- カンマ区切りで用語のみをリスト化
- 説明や定義は含めない
- 例: "アンモニア燃料エンジン, 舶用エンジン, レシプロエンジン"
- 関連用語がない場合は「なし」と出力"""

SIMILAR_TERMS_EXTRACTION_USER_PROMPT = """対象用語: {term}

以下の検索結果から、この用語と関連する専門用語を抽出してください。

検索結果:
{search_results}

関連する専門用語のみをカンマ区切りで出力してください。"""

CONTEXT_SUMMARY_SYSTEM_PROMPT = """あなたは技術文書の要約の専門家です。

**役割:**
- 検索結果から、対象用語に関連する重要な情報を抽出
- 冗長な情報を排除し、簡潔にまとめる
- 定義作成に必要な文脈を提供

**要約の原則:**
1. **関連性**: 対象用語に直接関連する情報のみ
2. **簡潔性**: 3〜5文程度にまとめる
3. **情報密度**: 重要な技術情報を優先
4. **構造化**: 読みやすく整理

**出力形式:**
- 段落形式または箇条書き
- 引用元の明示は不要
- 対象用語の文脈を明確にする"""

CONTEXT_SUMMARY_USER_PROMPT = """対象用語: {term}

以下の検索結果から、この用語に関連する重要な文脈を抽出・要約してください。

検索結果:
{search_results}

定義作成に役立つ簡潔な要約を出力してください。"""

RAG_FALLBACK_PROMPT = """提供されたコンテキストが不十分なため、一般的な知識に基づいて定義を作成します。

専門用語: {term}

可能な範囲で定義を作成してください。情報が不足している場合は、その旨を明記してください。"""

SYNONYM_DETECTION_SYSTEM_PROMPT = """あなたは専門用語の類義語判定の専門家です。

**役割:**
- 2つの専門用語とその定義を比較し、類義語かどうかを判定する
- 類義語の判定基準に基づいて、明確な理由とともに判断を示す

**類義語の判定基準:**
1. **完全同義**: 同じ概念を指す（例: CO2、二酸化炭素）
2. **ほぼ同義**: わずかな違いはあるが実質的に同じ（例: エンジン、機関）
3. **上位・下位概念**: 一方が他方を含む関係（例: 燃料エンジン ⊃ ディーゼルエンジン）
4. **関連語**: 関連はあるが類義語ではない（例: エンジン、燃料）
5. **無関係**: 関連性が薄い

**出力形式:**
以下のJSON形式で回答してください：
```json
{{
  "is_synonym": true/false,
  "relationship": "完全同義/ほぼ同義/上位概念/下位概念/関連語/無関係",
  "confidence": 0.0-1.0,
  "reason": "判定理由"
}}
```

**注意:**
- 類義語判定は厳密に行う
- 関連語と類義語を混同しない
- 上位・下位概念の場合は明示する"""

SYNONYM_DETECTION_USER_PROMPT = """以下の2つの専門用語が類義語かどうかを判定してください。

**用語1:** {term1}
**定義1:** {definition1}

**用語2:** {term2}
**定義2:** {definition2}

上記の判定基準に従って、JSON形式で回答してください。"""

CATEGORY_NAMING_SYSTEM_PROMPT = """あなたは専門用語グループの分類専門家です。

**役割:**
- 関連する専門用語とその定義を分析し、グループ全体を表す適切なカテゴリ名を生成する
- カテゴリ名は簡潔で理解しやすく、グループの本質を捉えたものにする

**カテゴリ名の原則:**
1. **簡潔性**: 2〜5単語程度の短いフレーズ
2. **包括性**: グループ全体の共通概念を表現
3. **明確性**: 専門家でない読者にも理解可能
4. **具体性**: 抽象的すぎず、適度に具体的

**出力形式:**
以下のJSON形式で回答してください：
```json
{{
  "category": "カテゴリ名",
  "confidence": 0.0-1.0,
  "reason": "このカテゴリ名を選んだ理由"
}}
```

**例:**
- グループ: [アンモニア, アンモニア燃料, アンモニアエンジン]
  → カテゴリ: "アンモニア関連技術"
- グループ: [ディーゼルエンジン, レシプロエンジン, 内燃機関]
  → カテゴリ: "内燃機関技術"
- グループ: [電動モーター, 電気モーター, 電動機]
  → カテゴリ: "電動駆動装置"
"""

CATEGORY_NAMING_USER_PROMPT = """以下の専門用語グループに、適切なカテゴリ名を付けてください。

**グループに含まれる専門用語と定義:**
{terms_with_definitions}

上記のグループ全体を表す、簡潔で適切なカテゴリ名をJSON形式で回答してください。"""

TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT = """あなたは辞書編纂の専門家として、収録すべき見出し語を選定する役割を担っています。

提示された用語について、辞書に収録する価値があるかを検討してください。

**収録の観点:**

辞書に載せるべき用語とは：
• 特定の分野で明確な意味を持つもの
• 識別子や記号として機能するもの
• 複数の概念が組み合わさった表現
• 専門的な活動や装置を表すもの
• 一般的でない外来語や新しい概念

辞書に載せる必要がない用語とは：
• 日常会話で頻繁に使われる基本的な単語
• 特別な意味を持たない単独の名詞
• どこでも同じ意味で使われる動作や状態を表す語

**参考となる判断例:**

収録すべきと判断された例：
• "XY-200形" - 特定の型を示す識別子として機能
• "窒素酸化物除去システム" - 複数の技術概念の組み合わせ
• "ISO14001" - 国際的な規格体系の識別子
• "バイオマス燃料" - 特定分野の複合概念
• "4サイクル方式" - 数値と技術の組み合わせ

収録不要と判断された例：
• "水" - 基本的な物質名
• "機械" - 広範囲を指す一般名詞
• "作る" - 日常的な動作
• "大きい" - 一般的な状態表現
• "計画" - どの分野でも同じ意味

**留意点:**
- 用語が持つ独自性や識別性を考慮
- 単独では一般的でも、組み合わせで特別な意味を持つ場合は収録対象
- 記号や数値を含む識別子は基本的に収録対象

**回答形式:**
```json
{{
  "is_technical": true/false,
  "confidence": 0.0-1.0,
  "reason": "選定理由の簡潔な説明"
}}
```
"""

TECHNICAL_TERM_JUDGMENT_USER_PROMPT = """以下の用語について、辞書への収録可否を検討してください。

**見出し語候補:** {term}

**説明文:**
{definition}

収録の価値があるかどうか、JSON形式でご回答ください。"""

CLUSTER_TERM_FILTERING_SYSTEM_PROMPT = """あなたは類似専門用語のフィルタリング専門家です。

**役割:**
- 同じカテゴリに属する類似した専門用語のグループから、辞書に残すべき用語を選択する
- 重複や冗長性を排除しつつ、独立した価値のある用語は残す

**判定基準:**

**残すべき用語:**
1. **独立した概念**: それぞれが異なる概念や技術を表す（例: 「エンジン」と「舶用エンジン」）
2. **異なる適用範囲**: 一般的な用語と特殊化された用語（例: 「アンモニア」と「アンモニア燃料」）
3. **高スコア用語**: スコアが高く、独立した専門用語として認識されている

**除外すべき用語:**
1. **単なる部分文字列**: 他の用語に完全に包含され、独自の意味がない
2. **スコアが著しく低い**: 他の用語のスコアの1/10以下など
3. **同義語・表記ゆれ**: 本質的に同じ意味（例: 「アンモニアガス」と「気体アンモニア」）

**重要な注意点:**
- スコアが同程度なら、それぞれ独立した専門用語として価値がある可能性が高い
- 定義の内容が明確に異なれば、両方残すべき
- 迷った場合は残す方向で判断

**出力形式:**
以下のJSON形式で回答してください：
```json
{{
  "keep_terms": ["用語1", "用語2"],
  "reason": "判定理由（各用語を残す/除外する根拠）"
}}
```
"""

CLUSTER_TERM_FILTERING_USER_PROMPT = """以下のカテゴリ「{category_name}」に属する類似用語グループから、辞書に残すべき用語を選択してください。

**用語グループ:**
{terms_info}

上記の判定基準に従って、残すべき用語をJSON形式で回答してください。"""


def get_definition_prompt_messages():
    """定義生成用のプロンプトメッセージを取得"""
    return [
        ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
        ("user", DEFINITION_GENERATION_USER_PROMPT)
    ]


def get_definition_prompt_messages_simple():
    """シンプル版定義生成プロンプト"""
    return [
        ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
        ("user", DEFINITION_GENERATION_USER_PROMPT_SIMPLE)
    ]


def get_similar_terms_prompt_messages():
    """類似用語抽出プロンプト"""
    return [
        ("system", SIMILAR_TERMS_EXTRACTION_SYSTEM_PROMPT),
        ("user", SIMILAR_TERMS_EXTRACTION_USER_PROMPT)
    ]


def get_context_summary_prompt_messages():
    """コンテキスト要約プロンプト"""
    return [
        ("system", CONTEXT_SUMMARY_SYSTEM_PROMPT),
        ("user", CONTEXT_SUMMARY_USER_PROMPT)
    ]


def get_synonym_detection_prompt_messages():
    """類義語判定プロンプト"""
    return [
        ("system", SYNONYM_DETECTION_SYSTEM_PROMPT),
        ("user", SYNONYM_DETECTION_USER_PROMPT)
    ]


def get_category_naming_prompt_messages():
    """カテゴリ名生成プロンプト"""
    return [
        ("system", CATEGORY_NAMING_SYSTEM_PROMPT),
        ("user", CATEGORY_NAMING_USER_PROMPT)
    ]


def get_technical_term_judgment_prompt_messages():
    """専門用語判定プロンプト"""
    return [
        ("system", TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT),
        ("user", TECHNICAL_TERM_JUDGMENT_USER_PROMPT)
    ]

def get_cluster_term_filtering_prompt_messages():
    """クラスタ内フィルタリングプロンプト"""
    return [
        ("system", CLUSTER_TERM_FILTERING_SYSTEM_PROMPT),
        ("user", CLUSTER_TERM_FILTERING_USER_PROMPT)
    ]


class PromptConfig:
    """プロンプト設定の管理クラス"""

    def __init__(self):
        self.definition_system = DEFINITION_GENERATION_SYSTEM_PROMPT
        self.definition_user = DEFINITION_GENERATION_USER_PROMPT
        self.definition_user_simple = DEFINITION_GENERATION_USER_PROMPT_SIMPLE
        self.similar_terms_system = SIMILAR_TERMS_EXTRACTION_SYSTEM_PROMPT
        self.similar_terms_user = SIMILAR_TERMS_EXTRACTION_USER_PROMPT
        self.context_summary_system = CONTEXT_SUMMARY_SYSTEM_PROMPT
        self.context_summary_user = CONTEXT_SUMMARY_USER_PROMPT
        self.fallback = RAG_FALLBACK_PROMPT

    def get_definition_prompt(self, use_similar_terms: bool = True):
        """定義生成プロンプトを取得"""
        if use_similar_terms:
            return get_definition_prompt_messages()
        else:
            return get_definition_prompt_messages_simple()

    def get_similar_terms_prompt(self):
        """類似用語抽出プロンプトを取得"""
        return get_similar_terms_prompt_messages()

    def get_context_summary_prompt(self):
        """コンテキスト要約プロンプトを取得"""
        return get_context_summary_prompt_messages()

    @classmethod
    def from_file(cls, filepath: str):
        """外部ファイルからプロンプトを読み込み（将来の拡張用）"""
        raise NotImplementedError("File-based prompt loading not yet implemented")