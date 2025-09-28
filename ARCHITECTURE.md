# 専門用語抽出システム V4 - アーキテクチャドキュメント

## 目次
1. [システム概要](#システム概要)
2. [全体アーキテクチャ](#全体アーキテクチャ)
3. [処理フロー詳細](#処理フロー詳細)
4. [パラメータ一覧](#パラメータ一覧)
5. [設計思想](#設計思想)
6. [V3からの改善点](#v3からの改善点)

---

## システム概要

### 目的
PDFや文書から専門用語を自動抽出し、定義を生成し、階層的にグループ化するシステム

### 主要技術
- **統計的手法**: TF-IDF, C-value
- **グラフアルゴリズム**: SemRe-Rank (意味的関連性グラフ + Personalized PageRank)
- **RAG**: BM25 + ベクトル検索による定義生成
- **LLM**: GPT-4oによる専門用語判定・カテゴリ命名
- **クラスタリング**: HDBSCAN による階層的類義語抽出

### 特徴
- **汎用性**: ドメイン特化のルールなし、統計的手法のみで抽出
- **正確性**: 論文準拠の正しいSemRe-Rank実装
- **柔軟性**: 各機能のON/OFF可能（定義生成、フィルタリング、階層化）

---

## 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      入力: PDF/テキスト                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: 候補語抽出（3手法統合）                               │
│ - 正規表現パターンマッチング（5パターン）                      │
│ - SudachiPy形態素解析 + 後方suffix生成                        │
│ - n-gram複合語生成（2~max_ngram）                            │
│ - 見出し特別処理                                              │
│ - 頻度フィルタリング                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: TF-IDF計算                                           │
│ - Term Frequency (TF)                                       │
│ - Inverse Document Frequency (IDF)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: C-value計算                                          │
│ - 複合語の重要度評価                                          │
│ - ネスト構造の考慮                                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 基底スコア統合: 0.7 × TF-IDF + 0.3 × C-value                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4-6: SemRe-Rank アルゴリズム                             │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ STEP 4: シード選定（エルボー法）                        │   │
│ │ - 基底スコア上位Z件から自動選定                         │   │
│ └───────────────────────────────────────────────────────┘   │
│                         ↓                                    │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ STEP 5: 意味的関連性グラフ構築                          │   │
│ │ - 埋め込みベクトル計算                                  │   │
│ │ - relmin (最小類似度) フィルタ                          │   │
│ │ - reltop (上位%) 選択                                  │   │
│ └───────────────────────────────────────────────────────┘   │
│                         ↓                                    │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ STEP 6: Personalized PageRank                          │   │
│ │ - シード用語から伝播                                    │   │
│ │ - ダンピングファクタ α = 0.85                          │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 最終スコア: base_score × PPR_score                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: RAG定義生成（オプション）                              │
│ - BM25キーワード検索                                          │
│ - ベクトル類似度検索                                          │
│ - LLM (GPT-4o) で定義生成                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: LLM専門用語判定（オプション）                          │
│ - 用語+定義をLLMに渡す                                        │
│ - 専門用語かどうか判定                                        │
│ - 一般用語でも独自定義なら専門用語                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 9: 階層的類義語抽出（オプション）                         │
│ - HDBSCAN密度ベースクラスタリング                             │
│ - condensed_tree から階層構造抽出                            │
│ - LLMでカテゴリ名生成                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               出力: 専門用語リスト + 階層構造                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 処理フロー詳細

### STEP 1: 候補語抽出

#### 目的
テキストから専門用語候補を最大限に抽出する（3つの手法を統合）

#### 処理内容

V4では**3つの抽出手法を統合**し、最大限の候補語を生成：
1. 正規表現パターンマッチング
2. SudachiPy形態素解析 + 後方suffix生成
3. n-gram複合語生成

##### 1.1 正規表現パターンマッチング

5つのパターンでマッチング：

```python
patterns = [
    r'[ァ-ヶー]+',                              # カタカナ
    r'[一-龯]{2,}',                             # 漢字（2文字以上）
    r'[0-9]*[A-Za-z][A-Za-z0-9]*',              # 英数字（数字始まり可）
    r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',    # カタカナ+漢字
    r'[一-龯]+[ァ-ヶー]+[一-龯]+',              # 漢字+カタカナ+漢字
]
```

**パターン詳細**:

1. **カタカナパターン**: `[ァ-ヶー]+`
   - マッチ例: "アンモニア", "エンジン", "システム"
   - 長音符「ー」も含む

2. **漢字パターン**: `[一-龯]{2,}`
   - マッチ例: "燃料", "機関", "削減目標"
   - 2文字以上に限定（単漢字を除外）

3. **英数字パターン**: `[0-9]*[A-Za-z][A-Za-z0-9]*`
   - マッチ例: "GHG", "6L28ADF", "CO2"
   - **重要**: 数字始まりを許可（"6L28ADF"全体を抽出）
   - 最低1つのアルファベット必須（純粋な数字を除外）

4. **混合パターン（カタカナ+漢字）**: `[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+`
   - マッチ例: "アンモニア燃料", "燃料エンジン"

5. **混合パターン（漢字+カタカナ+漢字）**: `[一-龯]+[ァ-ヶー]+[一-龯]+`
   - マッチ例: "国際エネルギー機関", "舶用アンモニア燃料"

##### 1.2 SudachiPy形態素解析

連続する名詞を連結し、自然な複合語を抽出：

```python
def _extract_candidates_with_sudachi(self, text: str) -> Dict[str, int]:
    tokens = self.sudachi_tokenizer.tokenize(text, self.sudachi_mode)

    current_phrase = []
    for token in tokens:
        pos = token.part_of_speech()[0]

        if pos in ['名詞', '接頭辞']:
            current_phrase.append(token.surface())
        else:
            if current_phrase:
                # 名詞句全体のみを追加（部分列生成はn-gramに任せる）
                phrase = ''.join(current_phrase)
                candidates[phrase] += 1
```

**例**:
- 入力: "舶用アンモニア燃料エンジン"
- 形態素解析: ["舶用", "アンモニア", "燃料", "エンジン"]
- 抽出結果:
  - "舶用アンモニア燃料エンジン" (全体のみ)

**設計方針**:
- 連続名詞の自然な区切りで複合語を抽出
- 部分列（"アンモニア燃料"など）はn-gramで生成するため重複を避ける

##### 1.3 n-gram複合語生成

形態素をn-gramで組み合わせ、あらゆる複合語候補を生成：

```python
def _generate_ngram_candidates(self, text: str) -> Dict[str, int]:
    tokens = self.sudachi_tokenizer.tokenize(text, self.sudachi_mode)
    nouns = [t.surface() for t in tokens if t.part_of_speech()[0] == '名詞']

    # 2-gram から max_ngram まで生成
    for window_size in range(2, min(self.max_ngram + 1, len(nouns) + 1)):
        for i in range(len(nouns) - window_size + 1):
            phrase = ''.join(nouns[i:i + window_size])
            candidates[phrase] += 1
```

**例**:
- 入力: "舶用アンモニア燃料エンジン開発"
- 名詞抽出: ["舶用", "アンモニア", "燃料", "エンジン", "開発"]
- n-gram生成 (max_ngram=3):
  - 2-gram: "舶用アンモニア", "アンモニア燃料", "燃料エンジン", "エンジン開発"
  - 3-gram: "舶用アンモニア燃料", "アンモニア燃料エンジン", "燃料エンジン開発"

##### 1.4 3手法の統合効果と役割分担

| 手法 | 抽出範囲 | 特徴 |
|------|---------|------|
| 正規表現 | 基本パターン | 高速、単純な文字種マッチング |
| SudachiPy | 自然な複合語全体 | 形態素解析による言語学的に正しい区切り |
| n-gram | あらゆる部分列 | 網羅性最大、前方・中間・後方すべてカバー |

**役割分担**:
- **正規表現**: 文字種の連続パターンを高速抽出
- **SudachiPy**: 形態素解析で自然な複合語の境界を判定（例: "舶用アンモニア燃料エンジン"を一つの単位として認識）
- **n-gram**: 複合語の部分列を体系的に生成（例: "舶用アンモニア", "アンモニア燃料", "燃料エンジン"）

**統合の理由**:
- 正規表現だけでは複合語の中間部分を抽出できない
- SudachiPyだけでは部分列を見逃す
- n-gramで最大限の候補を生成し、後段の統計的手法（TF-IDF/C-value/SemRe-Rank）で自然にフィルタリング
- **重複を避ける設計**: SudachiPyは全体のみ、n-gramで部分列生成を一元化

##### 1.5 数字のみフィルタ

```python
if re.match(r'^\d+$', term):
    continue  # "123" などは除外
```

##### 1.6 長さチェック

```python
if self.min_term_length <= len(term) <= self.max_term_length:
    candidates[term] += 1
```

- デフォルト: 2文字以上、15文字以下

##### 1.7 見出し特別処理

見出しから抽出された用語には以下の優遇措置：
- **出現頻度+2のボーナス**
- **頻度1でも残す**（重要な専門用語の可能性）

見出しパターン:
```python
heading_patterns = [
    r'^#{1,6}\s+(.+)$',                    # Markdown見出し
    r'^\d+\.\s+(.+)$',                     # 番号付き見出し
    r'^第[一二三四五六七八九十\d]+[章節項]\s*(.+)$',  # 章節見出し
]
```

##### 1.8 頻度フィルタリング

```python
if freq >= self.min_frequency:
    filtered[term] = freq
elif freq == 1 and term in heading_terms:
    filtered[term] = freq  # 見出し語は頻度1でも残す
```

#### 設計思想

**汎用性重視**:
- ❌ ストップワード除外なし
- ❌ 技術用語判定なし
- ❌ ドメイン特化ルールなし
- ✅ 統計的手法で自然にフィルタリング

**理由**:
- どんな文書にも対応可能
- 後段のTF-IDF/C-value/SemRe-Rankで自然にスコアリング
- LLMフィルタリングで最終判定

**網羅性重視**:
- 3つの抽出手法で最大限の候補を生成
- "過剰抽出 → 統計的フィルタ" のアプローチ
- 見逃しを最小化

---

### STEP 2: TF-IDF計算

#### 目的
用語の文書内重要度を統計的に評価

#### 数式

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = freq(t, d) / max_freq(d)

IDF(t) = log(N / df(t))
```

**記号**:
- `t`: 用語
- `d`: 文書
- `freq(t, d)`: 文書dにおける用語tの出現頻度
- `max_freq(d)`: 文書dの最大頻度
- `N`: 総文書数（文単位で分割）
- `df(t)`: 用語tが出現する文書数

#### 実装詳細

```python
def _calculate_tfidf(self, text: str, terms: List[str]) -> Dict[str, float]:
    # 文分割
    sentences = re.split(r'[。\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    # TF計算
    term_freq = Counter()
    for term in terms:
        term_freq[term] = len(re.findall(re.escape(term), text))

    max_freq = max(term_freq.values()) if term_freq else 1
    tf_scores = {term: freq / max_freq for term, freq in term_freq.items()}

    # IDF計算
    N = len(sentences)
    df = Counter()
    for sentence in sentences:
        unique_terms = set(term for term in terms if term in sentence)
        for term in unique_terms:
            df[term] += 1

    idf_scores = {term: math.log(N / count) if count > 0 else 0
                  for term, count in df.items()}

    # TF-IDF統合
    tfidf_scores = {term: tf_scores.get(term, 0) * idf_scores.get(term, 0)
                    for term in terms}

    return tfidf_scores
```

#### 特徴
- 頻出語は高TF
- 特定文にしか出ない語は高IDF
- TF-IDFが高い = 文書の特徴的な用語

---

### STEP 3: C-value計算

#### 目的
複合語の重要度を評価（ネスト構造を考慮）

#### 数式

```
C-value(a) = log₂|a| × freq(a)                    (aが他の用語に含まれない場合)

C-value(a) = log₂|a| × (freq(a) - 1/P(Ta) × Σ freq(b))  (含まれる場合)
                                                    b∈Ta
```

**記号**:
- `a`: 候補用語
- `|a|`: 用語aの文字数
- `freq(a)`: 用語aの出現頻度
- `Ta`: 用語aを含む長い用語の集合
- `P(Ta)`: Taのサイズ

#### 実装詳細

```python
def _calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
    cvalue_scores = {}
    term_list = list(candidates.keys())

    for term in term_list:
        freq = candidates[term]
        length = len(term)

        # ネストカウント: termを含む他の用語の数
        nested_count = 0
        for other_term in term_list:
            if term != other_term and term in other_term:
                nested_count += 1

        # C-value計算
        if nested_count == 0:
            # 独立用語
            cvalue = math.log(length + 1) * freq
        else:
            # ネストされた用語
            avg_freq = sum(candidates[t] for t in term_list
                          if term in t and term != t) / nested_count
            cvalue = math.log(length + 1) * (freq - (1.0 / nested_count) * avg_freq)

        cvalue_scores[term] = max(cvalue, 0.0)

    return cvalue_scores
```

#### 例

用語 | 頻度 | 長さ | ネスト | C-value計算
-----|------|------|--------|-------------
"アンモニア" | 10 | 5 | "アンモニア燃料"に含まれる | `log(6) × (10 - freq("アンモニア燃料")/1)`
"アンモニア燃料" | 5 | 7 | なし | `log(8) × 5`
"燃料" | 8 | 2 | "アンモニア燃料"に含まれる | `log(3) × (8 - 5/1)`

#### 特徴
- 長い複合語ほど高スコア（`log(length)`）
- 独立して出現する用語は高スコア
- 他の用語の一部としてのみ出現する用語は低スコア

---

### STEP 4-6: SemRe-Rank アルゴリズム

#### 概要
論文「SemRe-Rank」に基づく意味的関連性を考慮した専門用語抽出

#### STEP 4: シード選定（エルボー法）

##### 目的
Personalized PageRankの起点となるシード用語を自動選定

##### 処理フロー

```python
def _select_seeds(self, terms: List[str], base_scores: Dict[str, float]) -> List[str]:
    # 1. 基底スコアでソート
    sorted_terms = sorted(terms, key=lambda t: base_scores.get(t, 0), reverse=True)

    # 2. 上位Z件を候補
    top_z = sorted_terms[:self.seed_z]  # デフォルト: 50

    # 3. エルボー法でシード数決定
    scores = [base_scores.get(t, 0) for t in top_z]
    diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]

    # スコア差が急激に減少する点を検出
    elbow_idx = self.min_seed_count  # デフォルト: 5
    for i in range(self.min_seed_count, max_seeds):
        if diffs[i] < diffs[i-1] * 0.5:  # 差が半分以下になった点
            elbow_idx = i
            break

    # 4. シード用語を返す
    return top_z[:elbow_idx]
```

##### エルボー法の例

```
順位 | 用語              | スコア | 差分
-----|-------------------|--------|------
1    | アンモニア        | 2.63   | 1.51
2    | アンモニア燃料    | 1.12   | 0.29  ← 差分が急減
3    | 燃料              | 0.83   | 0.02
4    | エンジン          | 0.81   | 0.49
5    | GHG               | 0.32   | ...

→ シード: ["アンモニア", "アンモニア燃料"]
```

##### パラメータ

- `seed_z`: 候補数（デフォルト: 50）
- `min_seed_count`: 最小シード数（デフォルト: 5）
- `max_seed_ratio`: 最大シード比率（デフォルト: 0.7）
- `use_elbow_detection`: エルボー法使用（デフォルト: True）

---

#### STEP 5: 意味的関連性グラフ構築

##### 目的
用語間の意味的関連性をグラフとして表現

##### SemRe-Rank論文の定義

```
G = (V, E)

V: 全用語の集合
E: エッジの集合

エッジ (u, v) が存在する条件:
1. sim(u, v) >= relmin  (最小類似度閾値)
2. v ∈ TopK(u, reltop)  (uの上位reltop%の関連語)
```

##### 実装詳細

```python
def _build_semantic_graph(
    self,
    terms: List[str],
    embeddings: np.ndarray,
    seeds: List[str]
) -> nx.Graph:
    graph = nx.Graph()

    # 類似度行列計算
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    n = len(terms)

    # 各用語についてエッジを追加
    for i in range(n):
        graph.add_node(terms[i])

        # 他の全用語との類似度を計算
        sims = [(j, sim_matrix[i][j]) for j in range(n) if i != j]
        sims.sort(key=lambda x: x[1], reverse=True)

        # STEP 1: relmin閾値フィルタ
        sims = [(j, s) for j, s in sims if s >= self.relmin]

        # STEP 2: reltop上位選択
        top_k = max(1, int(len(sims) * self.reltop))
        sims = sims[:top_k]

        # エッジ追加
        for j, sim in sims:
            graph.add_edge(terms[i], terms[j], weight=float(sim))

    return graph
```

##### パラメータの意味

**relmin (最小類似度)**: デフォルト 0.5
- 類似度がこの値以上の用語ペアのみエッジを張る
- 高いほど厳格（エッジ数減少）
- 低いほど緩い（エッジ数増加）

**reltop (上位割合)**: デフォルト 0.15 (15%)
- 各用語について、relmin通過後の上位何%を選ぶか
- 高いほど密なグラフ
- 低いほど疎なグラフ

##### グラフ構造の例

```
用語: ["アンモニア", "アンモニア燃料", "燃料", "エンジン", "GHG"]

アンモニア
  ├─(0.87)→ アンモニア燃料
  └─(0.62)→ 燃料

アンモニア燃料
  ├─(0.87)→ アンモニア
  └─(0.73)→ 燃料

燃料
  ├─(0.73)→ アンモニア燃料
  └─(0.62)→ アンモニア

エンジン
  └─(0.54)→ GHG

GHG
  └─(0.54)→ エンジン
```

---

#### STEP 6: Personalized PageRank

##### 目的
シード用語から重要度を伝播させ、関連する専門用語を高く評価

##### PageRankアルゴリズム

標準PageRank:
```
PR(v) = (1-α)/N + α × Σ (PR(u) × w(u,v)) / Σ w(u,*)
                    u∈In(v)              u∈In(v)
```

Personalized PageRank:
```
PPR(v) = (1-α) × p(v) + α × Σ (PPR(u) × w(u,v)) / Σ w(u,*)
                              u∈In(v)              u∈In(v)

p(v) = { 1/|S|  if v ∈ S (シード)
       { 0      otherwise
```

**記号**:
- `v`: 対象ノード（用語）
- `α`: ダンピングファクタ（デフォルト: 0.85）
- `S`: シード用語集合
- `In(v)`: vへの入力エッジを持つノード
- `w(u,v)`: エッジ重み（コサイン類似度）
- `p(v)`: パーソナライゼーションベクトル

##### 実装詳細

```python
def _personalized_pagerank(
    self,
    graph: nx.Graph,
    seeds: List[str]
) -> Dict[str, float]:
    if len(graph) == 0:
        return {}

    # パーソナライゼーションベクトル初期化
    personalization = {}
    for node in graph.nodes():
        if node in seeds:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.0

    # 正規化
    total = sum(personalization.values())
    if total > 0:
        personalization = {k: v/total for k, v in personalization.items()}
    else:
        personalization = {node: 1.0/len(graph) for node in graph.nodes()}

    # PageRank実行
    try:
        ppr_scores = nx.pagerank(
            graph,
            alpha=self.alpha,
            personalization=personalization,
            max_iter=100,
            weight='weight'
        )
    except:
        ppr_scores = {node: 1.0 / len(graph) for node in graph.nodes()}

    return ppr_scores
```

##### PPRの特性

1. **シード用語は高スコア**
   - 初期値が1.0/|S|
   - 伝播を受け取る側でもある

2. **シードに近い用語も高スコア**
   - グラフ上で近い（高類似度エッジ）
   - 複数のシードから伝播を受ける

3. **孤立した用語は低スコア**
   - エッジが少ない
   - シードからの伝播が届かない

##### PPRスコア例

```
用語              | PPRスコア | 説明
------------------|----------|-----------------------------
アンモニア        | 0.25     | シード (1/4)
アンモニア燃料    | 0.25     | シード (1/4)
燃料              | 0.18     | 両シードから伝播
エンジン          | 0.12     | 弱い関連
GHG               | 0.10     | エンジン経由で伝播
```

---

#### 最終スコア統合

```python
final_scores = {}
for term in terms:
    base = base_scores.get(term, 0.01)
    ppr = ppr_scores.get(term, 0.5)
    final_scores[term] = base * ppr
```

**重要**: V3の加重和から**乗算**に変更

```
V3: final = γ×TF-IDF + β×C-value + w×PPR
V4: final = (0.7×TF-IDF + 0.3×C-value) × PPR
```

**理由**:
- SemRe-Rank論文の定義に準拠
- baseスコアとPPRは独立した評価軸
- 乗算により両方高い用語のみが高スコア

---

### STEP 7: RAG定義生成（オプション）

#### 目的
抽出された専門用語の定義を自動生成

#### アーキテクチャ

```
┌─────────────┐
│ 用語リスト   │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────┐
│ BM25キーワード検索                │
│ - 用語をクエリとして検索          │
│ - 上位K件のチャンク取得           │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│ ベクトル類似度検索                │
│ - 用語埋め込みでベクトル検索      │
│ - 上位K件のチャンク取得           │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│ 統合 (BM25 + Vector)             │
│ - 両方から取得したチャンクを統合  │
│ - 重複排除                        │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│ LLM定義生成 (GPT-4o)             │
│ - プロンプト: 用語+コンテキスト   │
│ - 構造化出力: 定義文              │
└─────────────────────────────────┘
```

#### 実装詳細

```python
def enrich_terms_with_definitions(
    terms: List[Term],
    text: str,
    top_n: int = 30,
    config: Optional[Config] = None,
    verbose: bool = True
) -> List[Term]:
    # PostgreSQL + PGVectorに接続
    vector_store = PGVector(
        collection_name="documents",
        connection_string=db_url,
        embedding_function=azure_embeddings
    )

    # BM25検索器を初期化
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 3

    # LCELチェーン構築
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store.as_retriever(search_kwargs={"k": 3})],
        weights=[0.5, 0.5]
    )

    for term in terms[:top_n]:
        # 関連チャンク取得
        docs = retriever.get_relevant_documents(term.term)
        context = "\n\n".join([doc.page_content for doc in docs])

        # LLM定義生成
        prompt = f"""
用語: {term.term}

以下の文脈から、この用語の定義を簡潔に説明してください。

文脈:
{context}

定義:
"""
        response = llm.invoke(prompt)
        term.definition = response.content.strip()

    return terms
```

#### プロンプト戦略

```python
DEFINITION_GENERATION_SYSTEM_PROMPT = """
あなたは専門用語の定義を生成する専門家です。

**定義の原則:**
1. **簡潔性**: 2〜3文で要点を説明
2. **正確性**: 文脈に基づいた正確な定義
3. **独立性**: その用語だけで理解できる説明
4. **専門性**: 技術的・専門的な観点を重視

**フォーマット:**
**[用語名]**
[定義文。具体的かつ簡潔に。]
"""
```

---

### STEP 8: LLM専門用語判定（オプション）

#### 目的
定義を考慮して、本当に専門用語かをLLMで判定

#### 判定基準

```python
TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT = """
あなたは専門用語判定の専門家です。

**専門用語と判定する場合:**
1. **技術用語**: 特定の技術分野に固有の用語
2. **業界用語**: 特定の業界で使われる専門的な用語
3. **独自定義**: 一般的な用語でも、その文脈で独自の専門的定義を持つ
4. **固有名詞**: 製品名、システム名、規格名など

**重要な注意点:**
- 用語名が一般的でも、**定義が専門的・独自的なら専門用語**
- 例: "状態" → 一般用語だが、"アイドル状態"なら専門用語

**専門用語でないと判定する場合:**
1. **一般名詞**: 日常的に使われる一般的な言葉
2. **抽象概念**: "こと"、"もの"、"ため" など
3. **時間表現**: "年"、"月"、"日" など
4. **数量表現**: "個"、"件"、"つ" など
"""
```

#### 実装

```python
def filter_technical_terms_by_definition(
    terms: List[Term],
    config: Optional[Config] = None,
    llm_model: str = "gpt-4o",
    verbose: bool = True
) -> List[Term]:
    technical_terms = []

    for term in terms:
        if not term.definition:
            # 定義なしは保留（スキップ）
            continue

        prompt = f"""
用語: {term.term}
定義: {term.definition}

この用語は専門用語ですか？

判定結果 (yes/no):
"""
        response = llm.invoke(prompt)
        judgment = response.content.strip().lower()

        if "yes" in judgment:
            technical_terms.append(term)

    return technical_terms
```

#### フィルタリング例

用語 | 定義 | 判定 | 理由
-----|------|------|------
"アンモニア" | "窒素と水素からなる無色の気体..." | YES | 化学物質名
"エンジン" | "燃料を燃焼させて動力を..." | YES | 技術用語
"状態" | "ある時点での様子や..." | NO | 一般名詞
"以上" | "それより大きいまたは..." | NO | 数量表現
"6L28ADF" | "船舶用ディーゼルエンジンの型式..." | YES | 固有名詞（製品型番）

---

### STEP 9: 階層的類義語抽出（オプション）

#### 目的
専門用語を意味的に類似したグループに階層的にクラスタリング

#### アルゴリズム: HDBSCAN

**HDBSCAN** = Hierarchical Density-Based Spatial Clustering of Applications with Noise

##### 特徴
- 密度ベースクラスタリング
- 階層構造を自動検出
- ノイズ（外れ値）を自動判定
- クラスタ数を事前指定不要

##### 処理フロー

```python
def extract_synonym_hierarchy(
    terms: List[Term],
    min_cluster_size: int = 2,
    generate_category_names: bool = True,
    use_umap: bool = False,  # オプション: UMAP次元削減
    umap_n_components: int = 50,
    verbose: bool = True
) -> Dict[str, SynonymHierarchy]:
    # 1. 定義埋め込み計算
    definitions = [t.definition for t in terms if t.definition]
    embeddings = azure_embeddings.embed_documents(definitions)
    embeddings = np.array(embeddings)  # shape: (n_terms, 1536)

    # 2. UMAP次元削減（オプション）
    if use_umap:
        reducer = UMAP(
            n_components=umap_n_components,  # 1536 → 50次元
            metric='cosine',
            random_state=42
        )
        embeddings = reducer.fit_transform(embeddings)

    # 3. HDBSCANクラスタリング
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # 4. 階層構造抽出
    condensed_tree = clusterer.condensed_tree_
    hierarchy = _build_hierarchy_from_tree(condensed_tree, terms, cluster_labels)

    # 5. 代表語選定
    for cluster_id, node in hierarchy.items():
        # クラスタの重心に最も近い用語を代表語とする
        cluster_embeddings = embeddings[node.term_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        representative_idx = np.argmin(distances)
        node.representative = node.terms[representative_idx]

    # 6. LLMでカテゴリ名生成
    if generate_category_names:
        for node in hierarchy.values():
            category_info = _generate_category_name(node, llm)
            node.category_name = category_info['name']
            node.category_confidence = category_info['confidence']
            node.category_reason = category_info['reason']

    return hierarchy
```

#### UMAP次元削減（オプション）

**目的**: 高次元埋め込み（1536次元）の次元の呪いを緩和

##### UMAPとは
- **UMAP** = Uniform Manifold Approximation and Projection
- 高次元データを低次元に圧縮しながら構造を保持
- t-SNEより高速で、大域的構造も保持

##### メリット
1. **次元の呪い緩和**: 1536次元 → 50次元
2. **クラスタリング精度向上**: 密度推定が安定
3. **計算コスト削減**: 距離計算が高速化

##### 使用場面
- 用語数が多い場合（50件以上）
- クラスタリング結果にノイズが多い場合
- より明確な階層構造を得たい場合

##### パラメータ
```python
use_umap=True,           # UMAP有効化
umap_n_components=50,    # 削減後の次元数（推奨: 30-100）
umap_metric='cosine'     # 距離メトリック（cosine推奨）
```

**注意**: UMAPは非決定的なため、`random_state=42`で再現性を確保

#### condensed_tree構造

```
HDBSCAN condensed_tree:

λ値    クラスタ    子ノード
----------------------------
0.5    cluster_0   [term1, term2, term3]
0.7    cluster_1   [term4, term5]
0.8    cluster_0   [cluster_1, term6]
```

- `λ`: 密度パラメータ（高いほど密）
- 階層構造を表現
- EOM (Excess of Mass) で最適クラスタ選択

#### カテゴリ名生成

```python
CATEGORY_NAMING_SYSTEM_PROMPT = """
あなたは専門用語グループの分類専門家です。

**カテゴリ名の原則:**
1. **簡潔性**: 2〜5単語程度の短いフレーズ
2. **包括性**: グループ全体の共通概念を表現
3. **識別性**: 他のグループと区別できる名称
4. **専門性**: 技術的・専門的な観点を重視

**出力形式:**
{
  "name": "カテゴリ名",
  "confidence": 0.95,
  "reason": "このグループは...という共通点があるため"
}
"""

def _generate_category_name(node: SynonymHierarchy, llm) -> dict:
    prompt = f"""
以下の専門用語グループのカテゴリ名を提案してください。

用語リスト:
{', '.join(node.terms)}

カテゴリ名:
"""
    response = llm.invoke(prompt)
    return json.loads(response.content)
```

#### クラスタリング例

```
クラスタ1: "アンモニア燃料技術"
  - アンモニア
  - アンモニア燃料
  - アンモニアガス
  - 燃料船
  - アンモニア燃料船
  信頼度: 0.95

クラスタ2: "燃料とエネルギー特性"
  - 燃料
  - 燃料全部
  - 燃料熱量
  - 代替燃料
  信頼度: 0.95

クラスタ3: "内燃機関と燃料技術"
  - エンジン
  - 発電機エンジン
  - ガスエンジン
  信頼度: 0.95
```

---

## パラメータ一覧

### 基本パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `min_term_length` | 2 | 最小用語長（文字数） |
| `max_term_length` | 15 | 最大用語長（文字数） |
| `min_frequency` | 2 | 最小出現頻度 |

### 候補語抽出パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_sudachi` | True | SudachiPy形態素解析を使用 |
| `use_ngram_generation` | True | n-gram複合語生成を使用 |
| `max_ngram` | 3 | n-gramの最大長 |

### SemRe-Rankパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `relmin` | 0.5 | 最小類似度閾値（0.0〜1.0） |
| `reltop` | 0.15 | 上位選択割合（0.0〜1.0） |
| `alpha` | 0.85 | PPRダンピングファクタ |
| `seed_z` | 50 | シード候補数 |
| `min_seed_count` | 5 | 最小シード数 |
| `max_seed_ratio` | 0.7 | 最大シード比率 |
| `use_elbow_detection` | True | エルボー法使用 |

### RAG/LLMパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enable_definition_generation` | True | 定義生成を有効化 |
| `enable_definition_filtering` | True | LLM判定を有効化 |
| `top_n_definition` | 30 | 定義生成する用語数 |
| `llm_model` | "gpt-4o" | 使用LLMモデル |

### 階層化パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enable_synonym_hierarchy` | True | 階層化を有効化 |
| `min_cluster_size` | 2 | HDBSCANの最小クラスタサイズ |
| `generate_category_names` | True | カテゴリ名を生成 |

---

## 設計思想

### 1. 汎用性優先

**原則**: ドメイン特化のルールを排除

- ❌ ストップワード除外
- ❌ 技術用語判定ルール
- ❌ 特定文字列の特別処理
- ✅ 統計的手法のみで抽出

**理由**:
- あらゆる文書に対応可能
- ルールメンテナンス不要
- 予期しない用語も抽出可能

### 2. 論文準拠の正確性

**SemRe-Rank論文との対応**:

| 論文の記述 | V4の実装 |
|-----------|---------|
| Base ATE (TF-IDF or C-value) | 0.7×TF-IDF + 0.3×C-value |
| Semantic graph with relmin/reltop | `_build_semantic_graph()` |
| Seed-based PPR | `_personalized_pagerank()` |
| Final = base × PPR | 乗算統合 |

### 3. モジュール性

各機能は独立してON/OFF可能:

```python
extractor = EnhancedTermExtractorV4(
    # 基本抽出のみ
    enable_definition_generation=False,
    enable_definition_filtering=False,
    enable_synonym_hierarchy=False
)

# または全機能有効
extractor = EnhancedTermExtractorV4(
    enable_definition_generation=True,
    enable_definition_filtering=True,
    enable_synonym_hierarchy=True
)
```

### 4. 統計 + LLMのハイブリッド

- **統計的手法**: 客観的・再現性高い
- **LLM**: 意味理解・柔軟な判定

役割分担:
- 統計: 候補抽出・スコアリング
- LLM: 定義生成・専門用語判定・カテゴリ命名

---

## V3からの改善点

### 1. 正しいSemRe-Rank実装

#### V3の問題

```python
# V3: kNNグラフ（固定k=12）
for i, term in enumerate(terms):
    distances = []
    for j, other_term in enumerate(terms):
        if i != j:
            sim = cosine_similarity(embeddings[i], embeddings[j])
            distances.append((j, sim))

    # 常に上位12件を選択
    distances.sort(key=lambda x: -x[1])
    for j, sim in distances[:12]:
        graph.add_edge(term, terms[j], weight=sim)
```

問題点:
- ❌ 論文のrelmin/reltop条件を無視
- ❌ 類似度に関係なく12件選択
- ❌ 疎な関係も密な関係も同じ扱い

#### V4の修正

```python
# V4: SemRe-Rankグラフ（relmin + reltop）
for i, term in enumerate(terms):
    sims = [(j, sim_matrix[i][j]) for j in range(n) if i != j]
    sims.sort(key=lambda x: x[1], reverse=True)

    # STEP 1: relmin閾値フィルタ
    sims = [(j, s) for j, s in sims if s >= self.relmin]

    # STEP 2: reltop上位選択
    top_k = max(1, int(len(sims) * self.reltop))
    sims = sims[:top_k]

    for j, sim in sims:
        graph.add_edge(terms[i], terms[j], weight=float(sim))
```

効果:
- ✅ 論文準拠の正確な実装
- ✅ 類似度に応じて適応的にエッジ数調整
- ✅ 疎なグラフ構造（不要なエッジ削減）

---

### 2. シード自動選定

#### V3の問題

```python
# V3: 手動またはランダムシード選定
seeds = ["アンモニア", "エンジン", "燃料"]  # 手動指定
```

問題点:
- ❌ 文書ごとに手動選定が必要
- ❌ 事前知識が必要
- ❌ 自動化困難

#### V4の修正

```python
# V4: エルボー法による自動選定
def _select_seeds(self, terms, base_scores):
    sorted_terms = sorted(terms, key=lambda t: base_scores[t], reverse=True)
    top_z = sorted_terms[:self.seed_z]

    scores = [base_scores[t] for t in top_z]
    diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]

    # スコア差の急減点を検出
    for i in range(self.min_seed_count, max_seeds):
        if diffs[i] < diffs[i-1] * 0.5:
            elbow_idx = i
            break

    return top_z[:elbow_idx]
```

効果:
- ✅ 完全自動化
- ✅ 文書に適応的
- ✅ 事前知識不要

---

### 3. スコア統合方法

#### V3の問題

```python
# V3: 複雑な加重和
final_score = γ * tfidf + β * cvalue + w * ppr

# パラメータチューニングが困難
γ = 0.4
β = 0.3
w = 0.3
```

問題点:
- ❌ 論文と異なる定義
- ❌ パラメータ調整が困難
- ❌ 各要素の独立性を無視

#### V4の修正

```python
# V4: 論文準拠の乗算
base_score = 0.7 * tfidf + 0.3 * cvalue
final_score = base_score * ppr
```

効果:
- ✅ SemRe-Rank論文に準拠
- ✅ シンプル
- ✅ 基底スコアとPPRが独立

---

### 4. 英数字パターン

#### V3の問題

```python
# V3: 大文字始まりのみ
r'[A-Z][A-Za-z0-9]*'

# "6L28ADF" → "L28ADF" （先頭の "6" が欠落）
```

#### V4の修正

```python
# V4: 数字始まり対応
r'[0-9]*[A-Za-z][A-Za-z0-9]*'

# "6L28ADF" → "6L28ADF" （完全に抽出）
# "123" → マッチしない（数字のみ除外）
```

効果:
- ✅ 型番を完全抽出
- ✅ 純粋な数字は除外

---

### 5. 候補語抽出の網羅性向上

#### V3の問題

```python
# V3: 正規表現のみ + SudachiPy後方suffix
candidates = extract_by_regex(text)  # 4パターン
candidates.update(extract_with_sudachi(text))  # 後方suffixのみ
```

問題点:
- ❌ 正規表現は基本パターンのみ（4パターン）
- ❌ 連続しない複合語を見逃す
- ❌ 形態素のn-gram組み合わせがない

**例**: "舶用アンモニア燃料エンジン開発"
- V3抽出: "舶用アンモニア燃料エンジン開発" (全体のみ)
- 見逃し: "舶用アンモニア", "アンモニア燃料", "燃料エンジン"

#### V4の修正

```python
# V4: 3手法統合（役割分担で重複削減）
# 1. 正規表現（5パターン）
candidates = extract_by_regex(text)

# 2. SudachiPy形態素解析（全体のみ、suffixなし）
if self.use_sudachi:
    candidates.update(extract_candidates_with_sudachi(text))

# 3. n-gram複合語生成（部分列を一元管理）
if self.use_ngram_generation:
    candidates.update(generate_ngram_candidates(text))
```

**例**: "舶用アンモニア燃料エンジン開発"
- V4抽出:
  - 正規表現: "舶用アンモニア燃料エンジン開発" (全体)
  - SudachiPy: "舶用アンモニア燃料エンジン開発" (全体のみ、重複)
  - n-gram: "舶用アンモニア", "アンモニア燃料", "燃料エンジン", "エンジン開発", "舶用アンモニア燃料", "アンモニア燃料エンジン", "燃料エンジン開発"

効果:
- ✅ 漢字+カタカナ+漢字パターン追加（5パターン）
- ✅ n-gramで部分列生成を一元化（前方・中間・後方すべてカバー）
- ✅ SudachiPyとn-gramの重複削減（suffixを削除）
- ✅ 見逃しを最小化
- ✅ 過剰抽出は統計的手法で自然にフィルタリング

---

### 6. 汎用性向上

#### V3の問題

```python
# V3: ストップワード・技術用語判定あり
stopwords = {"こと", "もの", "状態", "結果", ...}

if term in stopwords:
    continue

if not is_technical_term(term):
    continue
```

問題点:
- ❌ ドメイン特化
- ❌ メンテナンス必要
- ❌ 予期しない用語を除外

#### V4の修正

```python
# V4: ルールベース処理なし
# 統計的手法のみ
candidates = _extract_by_regex(text)
scores = calculate_tfidf_cvalue_semrerank(candidates)
```

効果:
- ✅ あらゆる文書に対応
- ✅ メンテナンスフリー
- ✅ 統計的手法で自然にフィルタリング

---

## まとめ

### V4の強み

1. **正確性**: SemRe-Rank論文に完全準拠
2. **汎用性**: ドメイン非依存、あらゆる文書に対応
3. **自動化**: シード選定からカテゴリ命名まで全自動
4. **柔軟性**: 各機能の独立したON/OFF
5. **拡張性**: RAG・LLM・クラスタリングの統合

### 推奨設定

**高精度優先**:
```python
EnhancedTermExtractorV4(
    min_frequency=2,
    relmin=0.6,          # 厳格
    reltop=0.1,          # 少数エッジ
    enable_definition_filtering=True
)
```

**高再現率優先**:
```python
EnhancedTermExtractorV4(
    min_frequency=1,
    relmin=0.4,          # 緩い
    reltop=0.2,          # 多数エッジ
    enable_definition_filtering=False
)
```

**バランス型**（デフォルト）:
```python
EnhancedTermExtractorV4(
    min_frequency=2,
    relmin=0.5,
    reltop=0.15,
    enable_definition_filtering=True
)
```

---

## 参考文献

1. **SemRe-Rank**: "SemRe-Rank: Improving Automatic Term Extraction by Incorporating Semantic Relatedness with Personalized PageRank"

2. **HDBSCAN**: "Density-Based Clustering Based on Hierarchical Density Estimates"

3. **TF-IDF**: "A Statistical Interpretation of Term Specificity and its Application in Retrieval"

4. **C-value**: "Automatic Recognition of Multi-Word Terms: the C-value/NC-value Method"

---

## 連絡先

GitHub: https://github.com/uchi736/Dictionary-of-Technical-Terms