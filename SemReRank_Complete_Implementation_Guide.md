# SemRe-Rank + LLMフィルタ 完全実装ガイド

## 概要

専門用語抽出の完全なパイプラインを解説します。

**処理フロー**:
1. **統計的抽出**: TF-IDF + C-value による候補抽出
2. **SemRe-Rank強化**: Personalized PageRankで意味的重要度を統合
3. **RAG定義生成**: ベクトルDBから文脈を取得して定義を自動生成
4. **LLMフィルタ**: 定義を基に専門用語か一般用語かを判定

**論文**: Zhang et al., 2017. "SemRe-Rank: Improving Automatic Term Extraction By Incorporating Semantic Relatedness With Personalised PageRank"

---

## 📊 完全シーケンス図

```mermaid
sequenceDiagram
    participant User
    participant System as EnhancedTermExtractor
    participant Stats as Statistics
    participant SRR as SemReRank
    participant RAG as RAG/VectorDB
    participant LLM as Azure OpenAI

    User->>System: extract(text)

    Note over System: Phase 1: 候補抽出
    System->>Stats: 候補用語抽出（n-gram/正規表現）
    Stats-->>System: candidates: {term: freq}

    Note over System: Phase 2: 基底スコア計算
    System->>Stats: TF-IDF計算
    Stats-->>System: tfidf_scores

    System->>Stats: C-value計算
    Stats-->>System: cvalue_scores

    System->>System: Stage A: 0.3×TF-IDF + 0.7×C-value<br/>(シード選定用)
    System->>System: Stage B: 0.7×TF-IDF + 0.3×C-value<br/>(最終スコア用)

    Note over System,SRR: Phase 3: SemRe-Rank強化
    System->>SRR: enhance_ate_scores(text, base_scores, seed_scores)

    SRR->>SRR: シード選定（エルボー法）
    SRR->>SRR: 単語抽出・埋め込み計算
    SRR->>SRR: 意味的関連性計算（コサイン類似度）

    loop 各文書（文単位）
        SRR->>SRR: グラフ構築（relmin=0.5, reltop=15%）
        SRR->>SRR: Personalized PageRank実行
        SRR->>SRR: 単語重要度を集約
    end

    SRR->>SRR: スコア改訂: (1 + avg_importance) × base_score
    SRR-->>System: enhanced_scores

    Note over System,RAG: Phase 4: RAG定義生成
    System->>RAG: 上位N件の用語について定義生成

    loop 各用語
        System->>RAG: similarity_search(term, k=5)
        RAG-->>System: 関連文書

        System->>LLM: 定義生成プロンプト<br/>(用語 + 文脈)
        LLM-->>System: definition
    end

    Note over System,LLM: Phase 5: LLMフィルタ
    System->>LLM: 専門用語判定（バッチ処理）

    loop 定義付き用語（バッチ）
        LLM->>LLM: 用語と定義を分析
        LLM->>LLM: 判定基準に基づき分類
        LLM-->>System: {is_technical, confidence, reason}
    end

    System->>System: 専門用語のみ抽出

    System-->>User: List[Term] (専門用語のみ)
```

---

## 🔑 核心ロジック詳細

### Phase 1-3: 統計抽出 + SemRe-Rank

**詳細は前半部分を参照（前述のガイド通り）**

---

### Phase 4: RAG定義生成

#### 4.1 ベクトル検索

```python
def enrich_terms_with_definitions(
    terms: List[Term],
    vector_store: PGVector,
    llm: AzureChatOpenAI,
    top_n: int = 30
) -> List[Term]:
    """
    上位N件の用語に定義を付与
    """
    for i, term in enumerate(terms[:top_n]):
        # 類似文書を検索（k=5件）
        docs = vector_store.similarity_search(
            query=term.term,
            k=5
        )

        # 文脈を抽出
        context = "\n\n".join([doc.page_content for doc in docs])

        # 定義生成プロンプト
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
            ("user", DEFINITION_GENERATION_USER_PROMPT_SIMPLE)
        ])

        chain = prompt | llm | StrOutputParser()

        # 定義生成
        definition = chain.invoke({
            "term": term.term,
            "context": context[:3000]  # トークン制限
        })

        term.definition = definition.strip()

    return terms
```

#### 定義生成プロンプト

```python
DEFINITION_GENERATION_SYSTEM_PROMPT = """あなたは専門用語の定義作成の専門家です。

**定義作成の原則:**
1. **簡潔性**: 1〜3文で定義を完結させる
2. **正確性**: 技術的に正確な情報のみを使用
3. **明確性**: 専門家でない読者にも理解できる表現
4. **コンテキスト**: 提供された文脈を活用
5. **構造化**: 必要に応じて箇条書きや段落分け

**出力形式:**
- 定義本文のみを出力
- 余計な前置きや締めくくりは不要
"""

DEFINITION_GENERATION_USER_PROMPT_SIMPLE = """以下の専門用語の定義を作成してください。

**専門用語:** {term}

**関連コンテキスト:**
{context}

上記の情報を基に、正確で理解しやすい定義を作成してください。"""
```

---

### Phase 5: LLMフィルタ（専門用語判定）

#### 5.1 判定ロジック

```python
def filter_technical_terms_by_definition(
    terms: List[Term],
    llm: AzureChatOpenAI
) -> List[Term]:
    """
    定義を基に専門用語をフィルタリング（バッチ処理）
    """
    # 定義がある用語のみ対象
    terms_with_def = [t for t in terms if t.definition]

    if not terms_with_def:
        return []

    # プロンプト準備
    prompt = ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT),
        ("user", TECHNICAL_TERM_JUDGMENT_USER_PROMPT)
    ])

    chain = prompt | llm | StrOutputParser()

    # バッチ処理で判定
    batch_inputs = [
        {"term": term.term, "definition": term.definition}
        for term in terms_with_def
    ]

    result_texts = chain.batch(batch_inputs)

    # 結果パース
    technical_terms = []
    for term, result_text in zip(terms_with_def, result_texts):
        result = _parse_judgment_result(result_text)

        if result and result.get("is_technical", False):
            technical_terms.append(term)
            print(f"  [OK] {term.term}: 専門用語 (信頼度: {result.get('confidence', 0):.2f})")
        else:
            print(f"  [NG] {term.term}: 一般用語")

    return technical_terms


def _parse_judgment_result(text: str) -> Optional[Dict]:
    """LLM判定結果のJSONをパース"""
    text = text.strip()

    # マークダウンコードブロック除去
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
```

---

#### 5.2 判定プロンプト

```python
TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT = """あなたは専門用語判定の専門家です。用語の形式とパターンから、専門用語か一般用語かを判定します。

**判定基準:**

【専門用語として判定】
• 型式番号・製品コード（6DE-18、L28ADFなど）
• 化学式・化合物名（CO2、NOx、アンモニアなど）
• 専門的な略語（GHG、MARPOL、IMOなど）
• 複合技術用語（アンモニア燃料エンジン、脱硝装置など）
• 数値+単位の仕様（2ストローク、50mg/kWhなど）
• 業界固有の用語（舶用エンジン、燃料噴射弁など）
• 規格・認証名（ISO14001、EIAPP証書など）

【一般用語として判定】
• 単体の基本名詞（ガス、燃料、エンジン、船など）
• 一般的な動詞・形容詞（使用、開発、最大、以上など）
• 抽象概念（目標、計画、状態、結果など）
• 日常用語（水、空気、時間、場所など）

**判定の例:**

専門用語の例：
• "6DE-18型エンジン" → ✓ 型式番号を含む
• "アンモニア燃料" → ✓ 複合技術用語
• "NOx排出量" → ✓ 化学式+技術用語
• "EIAPP証書" → ✓ 認証名
• "舶用ディーゼル" → ✓ 業界固有用語

一般用語の例：
• "ガス" → ✗ 単体の基本名詞
• "エンジン" → ✗ 単体の基本名詞
• "使用" → ✗ 一般動詞
• "最大" → ✗ 一般形容詞
• "目標" → ✗ 抽象概念

**重要:**
- 用語の形式・構造を重視
- 単体の一般名詞は原則除外
- 複合語や修飾語付きは専門用語の可能性が高い
- 型式番号、化学式、略語は専門用語として判定

**出力形式:**
```json
{
  "is_technical": true/false,
  "confidence": 0.0-1.0,
  "reason": "判定理由（簡潔に）"
}
```
"""

TECHNICAL_TERM_JUDGMENT_USER_PROMPT = """以下の用語とその定義を分析し、専門用語か一般用語かを判定してください。

**用語:** {term}

**定義:**
{definition}

上記の判定基準に従って、JSON形式で回答してください。"""
```

---

## 💻 完全な実装例

```python
from typing import List, Dict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from dictionary_system.core.models.base_extractor import Term

class CompleteTermExtractor:
    """完全な専門用語抽出パイプライン"""

    def __init__(
        self,
        db_url: str,
        collection_name: str = "documents",
        top_n_definition: int = 30
    ):
        self.top_n_definition = top_n_definition

        # Azure OpenAI
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small"
        )
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1-mini",
            temperature=0.0
        )

        # Vector Store
        self.vector_store = PGVector(
            collection_name=collection_name,
            connection=db_url,
            embeddings=self.embeddings
        )

    def extract(self, text: str) -> List[Term]:
        """完全な抽出パイプライン"""

        # Phase 1-2: 候補抽出 + 基底スコア
        candidates = self._extract_candidates(text)
        seed_scores = self._calculate_base_scores(text, candidates, stage="seed")
        base_scores = self._calculate_base_scores(text, candidates, stage="final")

        # Phase 3: SemRe-Rank強化
        enhanced_scores = self._apply_semrerank(
            text,
            base_scores,
            candidates,
            seed_scores
        )

        # Termオブジェクト化
        terms = [
            Term(term=term, score=score, definition="")
            for term, score in enhanced_scores.items()
        ]
        terms.sort(key=lambda x: x.score, reverse=True)

        # Phase 4: RAG定義生成（上位N件のみ）
        terms = self._enrich_with_definitions(terms)

        # Phase 5: LLMフィルタ（定義がある用語のみ）
        terms = self._filter_by_llm(terms)

        return terms

    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """候補用語抽出"""
        # n-gram生成、正規表現マッチング等
        pass

    def _calculate_base_scores(
        self,
        text: str,
        candidates: Dict[str, int],
        stage: str
    ) -> Dict[str, float]:
        """TF-IDF + C-value スコア計算"""

        tfidf_scores = self._calculate_tfidf(text, candidates)
        cvalue_scores = self._calculate_cvalue(candidates)

        # Min-max正規化（個別）
        tfidf_norm = self._min_max_normalize(tfidf_scores)
        cvalue_norm = self._min_max_normalize(cvalue_scores)

        # 段階別の重み設定
        if stage == "seed":
            w_tfidf, w_cvalue = 0.3, 0.7  # C-value重視
        else:
            w_tfidf, w_cvalue = 0.7, 0.3  # TF-IDF重視

        # 重み付き結合
        combined = {
            term: w_tfidf * tfidf_norm[term] + w_cvalue * cvalue_norm[term]
            for term in candidates
        }

        # 再正規化しない（重要！スコア0を避けるため）
        return combined

    def _apply_semrerank(
        self,
        text: str,
        base_scores: Dict[str, float],
        candidates: Dict[str, int],
        seed_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """SemRe-Rank適用"""
        from semrerank import SemReRank

        semrerank = SemReRank(
            use_azure_embeddings=True,
            relmin=0.5,
            reltop=0.15,
            alpha=0.85,
            seed_z=100,
            auto_select_seeds=True
        )

        return semrerank.enhance_ate_scores(
            text,
            base_scores,
            candidates,
            seed_selection_scores=seed_scores
        )

    def _enrich_with_definitions(self, terms: List[Term]) -> List[Term]:
        """RAG定義生成"""
        for i, term in enumerate(terms[:self.top_n_definition]):
            # ベクトル検索
            docs = self.vector_store.similarity_search(
                query=term.term,
                k=5
            )

            context = "\n\n".join([doc.page_content for doc in docs])

            # 定義生成
            prompt = ChatPromptTemplate.from_messages([
                ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
                ("user", DEFINITION_GENERATION_USER_PROMPT_SIMPLE)
            ])

            chain = prompt | self.llm | StrOutputParser()

            definition = chain.invoke({
                "term": term.term,
                "context": context[:3000]
            })

            term.definition = definition.strip()

        return terms

    def _filter_by_llm(self, terms: List[Term]) -> List[Term]:
        """LLM専門用語判定"""
        terms_with_def = [t for t in terms if t.definition]

        if not terms_with_def:
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system", TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT),
            ("user", TECHNICAL_TERM_JUDGMENT_USER_PROMPT)
        ])

        chain = prompt | self.llm | StrOutputParser()

        batch_inputs = [
            {"term": t.term, "definition": t.definition}
            for t in terms_with_def
        ]

        result_texts = chain.batch(batch_inputs)

        technical_terms = []
        for term, result_text in zip(terms_with_def, result_texts):
            result = self._parse_json(result_text)

            if result and result.get("is_technical", False):
                technical_terms.append(term)

        return technical_terms

    def _parse_json(self, text: str) -> Dict:
        """JSON応答のパース"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except:
            return {}

    def _min_max_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max正規化"""
        values = list(scores.values())
        min_val, max_val = min(values), max(values)

        if max_val - min_val < 1e-10:
            return {k: 0.5 for k in scores}

        return {
            term: (score - min_val) / (max_val - min_val)
            for term, score in scores.items()
        }

    def _calculate_tfidf(self, text: str, candidates: Dict[str, int]) -> Dict[str, float]:
        """TF-IDF計算（実装省略）"""
        pass

    def _calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """C-value計算（実装省略）"""
        pass
```

---

## 🎯 使用例

```python
# 初期化
extractor = CompleteTermExtractor(
    db_url="postgresql://user:pass@localhost:5432/mydb",
    collection_name="technical_docs",
    top_n_definition=30
)

# 抽出実行
text = """
アンモニア燃料エンジンは、次世代の環境対応技術として注目されている。
このアンモニア燃料エンジンは、従来のディーゼルエンジンと比較して、
CO2排出量を大幅に削減できる。6DE-18型エンジンでは、NOx排出量も
50mg/kWh以下に抑えられている。
"""

terms = extractor.extract(text)

# 結果表示
print("\n専門用語抽出結果:")
print("=" * 60)
for i, term in enumerate(terms, 1):
    print(f"{i:2}. {term.term:30} (Score: {term.score:.4f})")
    if term.definition:
        print(f"    定義: {term.definition[:80]}...")
```

**出力例**:
```
専門用語抽出結果:
============================================================
 1. アンモニア燃料エンジン               (Score: 4.2781)
    定義: アンモニアを燃料として使用するエンジン。CO2排出量が少なく、次世代の環境対応技術...
 2. 6DE-18型エンジン                  (Score: 2.1450)
    定義: 船舶用ディーゼルエンジンの型式。NOx排出量を50mg/kWh以下に抑える...
 3. NOx排出量                        (Score: 1.8920)
    定義: 窒素酸化物の排出量。環境規制の主要な指標の一つ...
 4. CO2排出量                        (Score: 1.6543)
    定義: 二酸化炭素の排出量。温室効果ガスの主要な指標...

[NG] ディーゼルエンジン: 一般用語
[NG] 環境: 一般用語
[NG] 技術: 一般用語
```

---

## 🔧 他システムへの組み込み

### パターン1: 既存RAGシステムへの統合

```python
class YourRAGSystem:
    def __init__(self):
        self.term_extractor = CompleteTermExtractor(
            db_url=self.db_url,
            collection_name="documents"
        )

    def index_document(self, document: str):
        """文書をインデックス化"""
        # 専門用語抽出
        terms = self.term_extractor.extract(document)

        # メタデータとして保存
        metadata = {
            "key_terms": [t.term for t in terms[:10]],
            "technical_terms": len(terms)
        }

        # ベクトルDBに保存
        self.vector_store.add_documents([
            Document(
                page_content=document,
                metadata=metadata
            )
        ])

    def retrieve_with_terms(self, query: str, k: int = 5):
        """クエリ拡張を使った検索"""
        # クエリから専門用語抽出
        query_terms = self.term_extractor.extract(query)

        # クエリ拡張
        expanded_query = f"{query} {' '.join([t.term for t in query_terms[:5]])}"

        # 検索
        return self.vector_store.similarity_search(expanded_query, k=k)
```

---

### パターン2: 用語辞書構築パイプライン

```python
class TermDictionaryBuilder:
    """専門用語辞書構築システム"""

    def __init__(self):
        self.extractor = CompleteTermExtractor(...)
        self.dictionary = {}

    def build_from_corpus(self, documents: List[str]):
        """コーパスから辞書構築"""
        all_terms = []

        for doc in documents:
            terms = self.extractor.extract(doc)
            all_terms.extend(terms)

        # 重複排除・統合
        self._merge_terms(all_terms)

    def _merge_terms(self, terms: List[Term]):
        """用語の統合・正規化"""
        term_dict = {}

        for term in terms:
            if term.term in term_dict:
                # スコアの平均
                existing = term_dict[term.term]
                existing.score = (existing.score + term.score) / 2
                # 定義の統合
                if not existing.definition and term.definition:
                    existing.definition = term.definition
            else:
                term_dict[term.term] = term

        self.dictionary = term_dict

    def export_to_json(self, filepath: str):
        """JSON形式でエクスポート"""
        data = [
            {
                "term": term.term,
                "definition": term.definition,
                "score": term.score,
                "metadata": term.metadata
            }
            for term in self.dictionary.values()
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
```

---

## ⚠️ 実装の注意点

### 1. スコア正規化の重要性（修正版）

```python
# ❌ 間違い: 結合後に再正規化する
combined = {
    term: 0.7 * tfidf_normalized[term] + 0.3 * cvalue_normalized[term]
    for term in candidates
}
return self._min_max_normalize(combined)  # これだと最小値が必ず0になる

# ✅ 正しい: 個別正規化のみ、結合後は正規化しない
tfidf_normalized = self._min_max_normalize(tfidf_scores)
cvalue_normalized = self._min_max_normalize(cvalue_scores)

combined = {
    term: 0.7 * tfidf_normalized[term] + 0.3 * cvalue_normalized[term]
    for term in candidates
}
return combined  # 再正規化しないことが重要
```

**理由**:
- min-max正規化は必ず最小値を0にする
- 結合後に再正規化すると、スコアが最小の用語が0になる
- `_revise_scores`で`base_score=0`の用語は最終スコアも0になる
- **解決策**: TF-IDFとC-valueを個別に正規化し、結合後はそのまま使用

---

### 2. LLMバッチ処理の最適化

```python
# バッチサイズを制限（トークン制限対策）
BATCH_SIZE = 10

for i in range(0, len(terms_with_def), BATCH_SIZE):
    batch = terms_with_def[i:i+BATCH_SIZE]
    batch_inputs = [
        {"term": t.term, "definition": t.definition}
        for t in batch
    ]
    result_texts = chain.batch(batch_inputs)
    # 処理...
```

---

### 3. エラーハンドリング

```python
def _parse_json(self, text: str) -> Dict:
    """堅牢なJSONパース"""
    try:
        # マークダウン除去
        text = re.sub(r'```json\s*|\s*```', '', text).strip()

        # JSON抽出（複数の方法を試す）
        try:
            return json.loads(text)
        except:
            # {}で囲まれた部分を抽出
            match = re.search(r'\{[^{}]*\}', text)
            if match:
                return json.loads(match.group())
            return {}

    except Exception as e:
        logger.warning(f"JSON parse error: {e}")
        return {}
```

---

## 📊 パフォーマンス最適化

### 1. キャッシュ戦略

```python
import hashlib
import pickle

class CachedExtractor(CompleteTermExtractor):
    def __init__(self, cache_dir="cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def extract(self, text: str) -> List[Term]:
        # キャッシュキー生成
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # キャッシュヒット
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # 通常処理
        terms = super().extract(text)

        # キャッシュ保存
        with open(cache_file, 'wb') as f:
            pickle.dump(terms, f)

        return terms
```

---

### 2. 並列処理

```python
from concurrent.futures import ThreadPoolExecutor

def extract_parallel(self, documents: List[str]) -> List[List[Term]]:
    """複数文書を並列処理"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(self.extract, documents))
    return results
```

---

## 📚 参考資料

- **論文**: Zhang et al., 2017. "SemRe-Rank"
- **LangChain**: https://python.langchain.com/
- **Azure OpenAI**: https://learn.microsoft.com/azure/ai-services/openai/
- **PGVector**: https://github.com/pgvector/pgvector

---

## ✅ 統合チェックリスト

実装時に確認:

### 統計抽出
- [ ] TF-IDFとC-valueの両方を計算
- [ ] 2段階スコアリング（Stage A: シード選定、Stage B: 最終）
- [ ] Min-max正規化を使用（個別のみ）
- [ ] **結合後のスコアは再正規化しない**（最重要）

### SemRe-Rank
- [ ] エルボー法でシード自動選定
- [ ] relmin=0.5, reltop=0.15で関連語フィルタ
- [ ] Personalized PageRankでパーソナライゼーション設定
- [ ] 文書レベルでスコア集約
- [ ] 式6でスコア改訂

### RAG定義生成
- [ ] ベクトルDBからk=5件の類似文書を取得
- [ ] 文脈を3000トークン以内に制限
- [ ] 定義生成プロンプトを適切に設定
- [ ] 上位N件のみ定義生成（コスト削減）

### LLMフィルタ
- [ ] 定義がある用語のみ判定対象
- [ ] バッチ処理で効率化
- [ ] 判定基準を明確に提示
- [ ] JSON出力をパース
- [ ] 信頼度スコアを記録

---

**作成日**: 2025-09-30
**バージョン**: 2.0 (LLMフィルタ統合版)
**対象システム**: RAG、専門用語抽出、辞書構築、文書分析