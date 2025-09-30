#!/usr/bin/env python3
"""
Synonym Extractor
==================
定義ベースの類義語抽出
- パターンA: 類似度フィルタ → LLM判定
- パターンB: HDBSCAN 階層的クラスタリング
"""

import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from dictionary_system.core.models.base_extractor import Term
from dictionary_system.config.prompts import (
    get_synonym_detection_prompt_messages,
    get_category_naming_prompt_messages
)
from dictionary_system.config.rag_config import Config


@dataclass
class SynonymPair:
    """類義語ペア"""
    term1: str
    term2: str
    relationship: str
    confidence: float
    reason: str
    similarity_score: float


class SynonymExtractor:
    """
    定義ベースの類義語抽出器

    処理フロー:
    1. 定義の埋め込みベクトルを計算
    2. コサイン類似度で候補ペアを抽出（閾値フィルタ）
    3. LLMで類義語かどうか判定
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        similarity_threshold: float = 0.7,
        max_candidates: int = 100,
        llm_model: str = "gpt-4.1-mini"
    ):
        """
        Args:
            config: 設定
            similarity_threshold: 類似度閾値（0.7以上を候補とする）
            max_candidates: 候補ペアの最大数
            llm_model: LLM モデル名
        """
        self.config = config or Config()
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_api_key,
            api_version=self.config.azure_openai_api_version,
            azure_deployment=self.config.azure_openai_embedding_deployment_name
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_api_key,
            api_version=self.config.azure_openai_api_version,
            azure_deployment=llm_model,
            temperature=0.0
        )

    def extract_synonyms(
        self,
        terms: List[Term],
        verbose: bool = True
    ) -> List[SynonymPair]:
        """
        類義語ペアを抽出

        Args:
            terms: 定義付き用語リスト
            verbose: 進捗表示

        Returns:
            類義語ペアのリスト
        """
        terms_with_def = [t for t in terms if t.definition]

        if len(terms_with_def) < 2:
            return []

        if verbose:
            print(f"類義語抽出: {len(terms_with_def)}件の用語を処理中...")

        definition_embeddings = self._compute_embeddings(terms_with_def)

        candidates = self._find_candidate_pairs(
            terms_with_def,
            definition_embeddings,
            verbose=verbose
        )

        if not candidates:
            return []

        if verbose:
            print(f"候補ペア: {len(candidates)}件")
            print("LLMで類義語判定中...")

        synonyms = self._judge_synonyms_with_llm(
            candidates,
            verbose=verbose
        )

        if verbose:
            print(f"類義語ペア: {len(synonyms)}件")

        return synonyms

    def _compute_embeddings(self, terms: List[Term]) -> np.ndarray:
        """定義の埋め込みベクトルを計算"""
        definitions = [t.definition for t in terms]
        embeddings_list = self.embeddings.embed_documents(definitions)
        return np.array(embeddings_list)

    def _find_candidate_pairs(
        self,
        terms: List[Term],
        embeddings: np.ndarray,
        verbose: bool = True
    ) -> List[Tuple[Term, Term, float]]:
        """類似度でペア候補を抽出"""
        candidates = []

        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                similarity = self._cosine_similarity(
                    embeddings[i],
                    embeddings[j]
                )

                if similarity >= self.similarity_threshold:
                    candidates.append((terms[i], terms[j], similarity))

        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:self.max_candidates]

        return candidates

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度を計算"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def _judge_synonyms_with_llm(
        self,
        candidates: List[Tuple[Term, Term, float]],
        verbose: bool = True
    ) -> List[SynonymPair]:
        """LLMで類義語判定"""
        prompt = ChatPromptTemplate.from_messages(
            get_synonym_detection_prompt_messages()
        )

        chain = prompt | self.llm | StrOutputParser()

        synonyms = []

        for i, (term1, term2, sim_score) in enumerate(candidates, 1):
            if verbose and i % 10 == 0:
                print(f"  {i}/{len(candidates)}件処理中...")

            result_text = chain.invoke({
                "term1": term1.term,
                "definition1": term1.definition,
                "term2": term2.term,
                "definition2": term2.definition
            })

            result = self._parse_llm_result(result_text)

            if result and result.get("is_synonym", False):
                synonyms.append(SynonymPair(
                    term1=term1.term,
                    term2=term2.term,
                    relationship=result.get("relationship", "不明"),
                    confidence=result.get("confidence", 0.0),
                    reason=result.get("reason", ""),
                    similarity_score=sim_score
                ))

        return synonyms

    def _parse_llm_result(self, text: str) -> Optional[Dict]:
        """LLM結果のJSONをパース"""
        text = text.strip()

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


def extract_synonym_groups(
    terms: List[Term],
    config: Optional[Config] = None,
    similarity_threshold: float = 0.7,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    類義語グループを抽出

    Args:
        terms: 定義付き用語リスト
        config: 設定
        similarity_threshold: 類似度閾値
        verbose: 進捗表示

    Returns:
        {代表語: [類義語リスト]} の辞書
    """
    extractor = SynonymExtractor(
        config=config,
        similarity_threshold=similarity_threshold
    )

    pairs = extractor.extract_synonyms(terms, verbose=verbose)

    groups = _build_synonym_groups(pairs)

    return groups


def _build_synonym_groups(pairs: List[SynonymPair]) -> Dict[str, List[str]]:
    """類義語ペアからグループを構築"""
    from collections import defaultdict

    graph = defaultdict(set)

    for pair in pairs:
        graph[pair.term1].add(pair.term2)
        graph[pair.term2].add(pair.term1)

    visited = set()
    groups = {}

    def dfs(node, group):
        if node in visited:
            return
        visited.add(node)
        group.add(node)
        for neighbor in graph[node]:
            dfs(neighbor, group)

    for node in graph:
        if node not in visited:
            group = set()
            dfs(node, group)

            representative = sorted(group)[0]
            groups[representative] = sorted(group)

    return groups


@dataclass
class SynonymHierarchy:
    """階層的類義語構造"""
    representative: str
    terms: List[str]
    category_name: Optional[str] = None
    category_confidence: float = 0.0
    category_reason: str = ""
    children: Dict[str, 'SynonymHierarchy'] = field(default_factory=dict)
    level: int = 0
    cluster_id: int = -1


class HierarchicalSynonymExtractor:
    """
    HDBSCAN による階層的類義語抽出

    処理フロー:
    1. 定義埋め込みで HDBSCAN クラスタリング
    2. condensed_tree から階層構造抽出
    3. 各クラスタの代表語を選定
    4. 階層ツリーを構築
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        llm_model: str = "gpt-4.1-mini",
        use_umap: bool = False,
        umap_n_components: int = 50,
        umap_metric: str = "cosine"
    ):
        """
        Args:
            config: 設定
            min_cluster_size: 最小クラスタサイズ
            min_samples: HDBSCAN の min_samples
            llm_model: カテゴリ名生成用LLMモデル
            use_umap: UMAP次元削減を使用するか
            umap_n_components: UMAP削減後の次元数（デフォルト: 50）
            umap_metric: UMAP距離メトリック（cosine推奨）
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan not available. pip install hdbscan")

        self.config = config or Config()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric

        if self.use_umap and not UMAP_AVAILABLE:
            raise ImportError("umap-learn not available. pip install umap-learn")

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_api_key,
            api_version=self.config.azure_openai_api_version,
            azure_deployment=self.config.azure_openai_embedding_deployment_name
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_api_key,
            api_version=self.config.azure_openai_api_version,
            azure_deployment=llm_model,
            temperature=0.0
        )

    def extract_hierarchy(
        self,
        terms: List[Term],
        generate_category_names: bool = True,
        verbose: bool = True
    ) -> Dict[str, SynonymHierarchy]:
        """
        階層的類義語構造を抽出

        Args:
            terms: 定義付き用語リスト
            generate_category_names: LLMでカテゴリ名を生成するか
            verbose: 進捗表示

        Returns:
            {代表語: SynonymHierarchy} の辞書
        """
        terms_with_def = [t for t in terms if t.definition]

        if len(terms_with_def) < self.min_cluster_size:
            return {}

        if verbose:
            print(f"HDBSCAN 階層的クラスタリング: {len(terms_with_def)}件")

        definition_embeddings = self._compute_embeddings(terms_with_def)

        if self.use_umap:
            if verbose:
                print(f"  UMAP次元削減: {definition_embeddings.shape[1]} → {self.umap_n_components}次元")
            
            reducer = UMAP(
                n_components=self.umap_n_components,
                metric=self.umap_metric,
                random_state=42
            )
            definition_embeddings = reducer.fit_transform(definition_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        clusterer.fit(definition_embeddings)

        if verbose:
            n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
            n_noise = list(clusterer.labels_).count(-1)
            print(f"  クラスタ数: {n_clusters}")
            print(f"  ノイズ: {n_noise}件")

        hierarchy = self._build_hierarchy(
            terms_with_def,
            clusterer,
            verbose=verbose
        )

        if generate_category_names and hierarchy:
            if verbose:
                print(f"カテゴリ名生成中...")
            self._generate_category_names(
                hierarchy,
                terms_with_def,
                verbose=verbose
            )

        return hierarchy

    def _compute_embeddings(self, terms: List[Term]) -> np.ndarray:
        """定義の埋め込みベクトルを計算"""
        definitions = [t.definition for t in terms]
        embeddings_list = self.embeddings.embed_documents(definitions)
        return np.array(embeddings_list)

    def _build_hierarchy(
        self,
        terms: List[Term],
        clusterer,
        verbose: bool = True
    ) -> Dict[str, SynonymHierarchy]:
        """階層構造を構築（HDBSCANのcondensed_tree + 用語名の包含関係）"""
        from collections import defaultdict
        import logging
        logger = logging.getLogger(__name__)

        # 葉ノードのクラスタリング結果
        cluster_terms = defaultdict(list)
        term_to_cluster = {}
        
        for i, label in enumerate(clusterer.labels_):
            if label != -1:
                cluster_terms[label].append(terms[i])
                term_to_cluster[terms[i].term] = label

        # 葉レベルのクラスタを作成
        hierarchies = {}
        cluster_id_to_node = {}

        for cluster_id, cluster_terms_list in cluster_terms.items():
            representative = self._select_representative(cluster_terms_list)

            hierarchy = SynonymHierarchy(
                representative=representative,
                terms=[t.term for t in cluster_terms_list],
                cluster_id=cluster_id,
                level=0
            )

            hierarchies[representative] = hierarchy
            cluster_id_to_node[cluster_id] = hierarchy

        # STEP 1: condensed_treeからHDBSCANの階層を構築
        if hasattr(clusterer, 'condensed_tree_') and len(cluster_id_to_node) > 1:
            logger.info(f"STEP 1: HDBSCAN階層抽出開始 ({len(cluster_id_to_node)}個のクラスタ)")
            self._extract_hdbscan_hierarchy(
                clusterer.condensed_tree_,
                cluster_id_to_node,
                hierarchies,
                term_to_cluster,
                verbose=verbose
            )
        else:
            if not hasattr(clusterer, 'condensed_tree_'):
                logger.warning(f"condensed_tree_が存在しません")
            elif len(cluster_id_to_node) <= 1:
                logger.info(f"階層構築スキップ: クラスタ数が1以下 ({len(cluster_id_to_node)})")
        
        # STEP 2: 用語名の包含関係で階層を補強
        logger.info(f"STEP 2: 用語包含関係で階層補強")
        self._enrich_with_subsumption(hierarchies, verbose=verbose)

        return hierarchies

    def _extract_hdbscan_hierarchy(
        self,
        condensed_tree,
        cluster_id_to_node: Dict[int, SynonymHierarchy],
        hierarchies: Dict[str, SynonymHierarchy],
        term_to_cluster: Dict[str, int],
        verbose: bool = True
    ):
        """
        HDBSCANのcondensed_treeから真の階層を抽出
        
        condensed_treeの構造:
        - parent: 親ノードID（内部ノードは大きい値）
        - child: 子ノードID（クラスタIDは小さい値 0,1,2...）
        - lambda_val: 分離の強さ
        - child_size: 子のサイズ
        
        階層構築の方針:
        1. 最も密接に関連するクラスタを見つける（同じ親から分離）
        2. それらを親ノードの下にグループ化
        3. 親ノードには代表用語を割り当て
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            tree_data = condensed_tree._raw_tree
            cluster_ids = set(cluster_id_to_node.keys())
            
            logger.info(f"condensed_tree: {len(tree_data)}行, クラスタ数={len(cluster_ids)}")
            
            # 親ノードIDごとに子クラスタをグループ化
            parent_to_clusters = defaultdict(list)
            
            for row in tree_data:
                parent = int(row['parent'])
                child = int(row['child'])
                lambda_val = float(row['lambda_val'])
                
                # 子が最終クラスタの場合
                if child in cluster_ids:
                    parent_to_clusters[parent].append({
                        'cluster_id': child,
                        'lambda': lambda_val
                    })
            
            # 複数のクラスタを持つ親ノードのみ処理
            parent_nodes_created = 0
            
            for parent_id, children_info in parent_to_clusters.items():
                if len(children_info) < 2:
                    continue  # 1つしか子がない親はスキップ
                
                # 子クラスタのノードを取得
                child_cluster_ids = [c['cluster_id'] for c in children_info]
                child_nodes = [cluster_id_to_node[cid] for cid in child_cluster_ids]
                
                # 親ノードの代表語：子クラスタの用語から選ぶ
                all_child_terms = []
                for child_node in child_nodes:
                    all_child_terms.extend(child_node.terms)
                
                if not all_child_terms:
                    continue
                
                # 最も短い用語を親の代表語にする
                parent_representative = min(all_child_terms, key=len)
                
                # 親ノードを作成
                parent_node = SynonymHierarchy(
                    representative=parent_representative,
                    terms=[],  # 親ノードは直接用語を持たない
                    cluster_id=parent_id,
                    level=1,
                    category_name=f"{parent_representative}系"
                )
                
                # 子ノードを親に追加
                for child_node in child_nodes:
                    parent_node.children[child_node.representative] = child_node
                    child_node.level = 1
                    
                    # 子をトップレベルから削除
                    if child_node.representative in hierarchies:
                        del hierarchies[child_node.representative]
                
                # 親をトップレベルに追加
                hierarchies[parent_node.representative] = parent_node
                parent_nodes_created += 1
                
                logger.info(f"  親ノード作成: {parent_representative} ({len(child_nodes)}個の子クラスタ)")
            
            if parent_nodes_created > 0:
                logger.info(f"階層構造: {parent_nodes_created}個の親ノードを生成")
            else:
                logger.info(f"階層構造: 親ノードなし（全クラスタが独立）")
                
        except Exception as e:
            logger.error(f"condensed_treeからの階層抽出失敗: {e}", exc_info=True)

    def _enrich_with_subsumption(
        self,
        hierarchies: Dict[str, SynonymHierarchy],
        verbose: bool = True
    ):
        """
        用語名の包含関係で階層を補強
        
        既存の階層に対して、用語名の包含関係を追加で検出し、
        意味のある親子関係があれば階層に反映する。
        
        例: 「エンジン」クラスタと「アンモニア燃料エンジン」クラスタがあれば、
            包含関係を検出して親子にする
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 全ノード（親・子含む）をフラット化
        all_nodes = {}
        
        def collect_nodes(nodes_dict):
            """再帰的に全ノードを収集"""
            for rep, node in nodes_dict.items():
                all_nodes[rep] = node
                if node.children:
                    collect_nodes(node.children)
        
        collect_nodes(hierarchies)
        
        # 各ノードの全用語をリストアップ
        node_terms = {}
        for rep, node in all_nodes.items():
            node_terms[rep] = set(node.terms)
            # 子ノードの用語も含める
            for child_node in node.children.values():
                node_terms[rep].update(child_node.terms)
        
        # ノード間の包含関係を検出
        subsumption_pairs = []  # (parent_rep, child_rep, score)
        
        for i, (rep1, terms1) in enumerate(node_terms.items()):
            for j, (rep2, terms2) in enumerate(node_terms.items()):
                if i >= j:
                    continue
                
                # どちらかの用語が他方を包含しているか
                for term1 in terms1:
                    for term2 in terms2:
                        if term1 != term2:
                            # term2がterm1を包含 → term1が親
                            if term1 in term2 and len(term1) < len(term2):
                                subsumption_pairs.append((rep1, rep2, len(term2) - len(term1)))
                            # term1がterm2を包含 → term2が親
                            elif term2 in term1 and len(term2) < len(term1):
                                subsumption_pairs.append((rep2, rep1, len(term1) - len(term2)))
        
        if not subsumption_pairs:
            logger.info("  用語包含関係: 検出なし")
            return
        
        # スコアでソート（包含の差が大きいものを優先）
        subsumption_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 包含関係を階層に反映
        added_count = 0
        
        for parent_rep, child_rep, score in subsumption_pairs:
            parent_node = all_nodes.get(parent_rep)
            child_node = all_nodes.get(child_rep)
            
            if not parent_node or not child_node:
                continue
            
            # 既に親子関係がある場合はスキップ
            if child_rep in parent_node.children:
                continue
            
            # 逆の関係（子が親を含む）がある場合はスキップ
            if parent_rep in child_node.children:
                continue
            
            # 同じレベルにある場合のみ親子関係を追加
            # （既存の階層を壊さないため）
            if parent_node.level == child_node.level:
                # 親ノードに子を追加
                parent_node.children[child_rep] = child_node
                child_node.level = parent_node.level + 1
                
                # 子ノードをトップレベルから削除（親の下に移動）
                if child_rep in hierarchies:
                    del hierarchies[child_rep]
                
                added_count += 1
                logger.info(f"  包含関係追加: {parent_rep} -> {child_rep}")
        
        if added_count > 0:
            logger.info(f"  用語包含関係: {added_count}個の親子関係を追加")
        else:
            logger.info(f"  用語包含関係: 追加可能な関係なし")

    def _extract_subsumption_hierarchy(
        self,
        hierarchies: Dict[str, SynonymHierarchy],
        verbose: bool = True
    ):
        """
        用語名の包含関係から階層を構築
        
        例: 「エンジン」⊂「アンモニア燃料エンジン」⊂「6DE-28エンジン」
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 全用語をフラット化
        all_terms = []
        term_to_cluster = {}
        
        for rep, node in list(hierarchies.items()):
            for term in node.terms:
                all_terms.append(term)
                term_to_cluster[term] = node
        
        # 用語間の包含関係を検出
        parent_child_map = {}  # child_term -> parent_term
        
        for i, child_term in enumerate(all_terms):
            for j, parent_term in enumerate(all_terms):
                if i != j and parent_term != child_term:
                    # child_termがparent_termを含む場合、parent_termが上位概念
                    if parent_term in child_term and len(parent_term) < len(child_term):
                        # より短い親が既に存在する場合はスキップ（最も近い親のみ）
                        if child_term not in parent_child_map or len(parent_term) > len(parent_child_map[child_term]):
                            parent_child_map[child_term] = parent_term
        
        if not parent_child_map:
            logger.info("用語包含関係: 階層関係なし（全て独立クラスタ）")
            return
        
        logger.info(f"用語包含関係: {len(parent_child_map)}個の親子関係を検出")
        
        # クラスタレベルの親子関係を構築
        cluster_parent_child = {}  # child_cluster -> parent_cluster
        
        for child_term, parent_term in parent_child_map.items():
            child_cluster = term_to_cluster[child_term]
            parent_cluster = term_to_cluster[parent_term]
            
            if child_cluster != parent_cluster:
                # 異なるクラスタ間の関係のみ
                if child_cluster.representative not in cluster_parent_child:
                    cluster_parent_child[child_cluster.representative] = parent_cluster.representative
        
        if not cluster_parent_child:
            logger.info("クラスタ間階層: 関係なし（クラスタ内包含のみ）")
            return
        
        logger.info(f"クラスタ間階層: {len(cluster_parent_child)}個のクラスタ親子関係")
        
        # 階層構造を再構築
        for child_rep, parent_rep in cluster_parent_child.items():
            if child_rep in hierarchies and parent_rep in hierarchies:
                parent_node = hierarchies[parent_rep]
                child_node = hierarchies[child_rep]
                
                # 親ノードに子を追加
                parent_node.children[child_rep] = child_node
                child_node.level = parent_node.level + 1
                
                # 子ノードをトップレベルから削除
                if child_rep in hierarchies:
                    del hierarchies[child_rep]
                
                logger.info(f"  {parent_rep} -> {child_rep}")

    def _extract_parent_child_relationships(
        self,
        condensed_tree,
        cluster_id_to_node: Dict[int, SynonymHierarchy],
        hierarchies: Dict[str, SynonymHierarchy],
        verbose: bool = True
    ):
        """
        condensed_treeから親子関係を抽出

        HDBSCANのcondensed_treeは階層的なクラスタマージ情報を持つ：
        - parent: 親クラスタID (内部ノードID、通常は n_samples 以上)
        - child: 子クラスタID (葉は n_samples未満、内部ノードは以上)
        - lambda_val: 分離の強さ
        """
        try:
            # condensed_tree._raw_treeからマージ情報を取得
            tree_data = condensed_tree._raw_tree

            # クラスタIDセット（葉レベル）
            cluster_ids = set(cluster_id_to_node.keys())

            # 内部ノードIDの範囲を推定（通常はn_samples以上）
            # condensed_treeでは、クラスタIDは小さい値（0,1,2...）
            # 内部ノードIDは大きい値（n_samples以上）
            n_samples = max(max(row['parent'], row['child']) for row in tree_data) + 1

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"condensed_tree: {len(tree_data)}行, クラスタID={cluster_ids}")

            # 親子関係のマッピング（内部ノードIDで）
            parent_to_children = defaultdict(list)
            child_to_parent = {}
            internal_node_to_clusters = defaultdict(set)

            # まず、内部ノード→クラスタの対応を構築
            for row in tree_data:
                parent = int(row['parent'])
                child = int(row['child'])

                # 子がクラスタIDの場合
                if child in cluster_ids:
                    internal_node_to_clusters[parent].add(child)
                    child_to_parent[child] = parent

                # 親→子の関係を記録（内部ノード間も含む）
                parent_to_children[parent].append(child)

            # 再帰的に内部ノードの所属クラスタを伝播
            def get_all_clusters(node_id):
                """内部ノードが含む全クラスタIDを取得（再帰）"""
                if node_id in cluster_ids:
                    return {node_id}

                clusters = set()
                if node_id in internal_node_to_clusters:
                    clusters.update(internal_node_to_clusters[node_id])

                for child in parent_to_children.get(node_id, []):
                    clusters.update(get_all_clusters(child))

                return clusters

            # 親クラスタを作成（複数のクラスタを含む内部ノード）
            created_parents = set()

            for parent_node_id, child_node_ids in parent_to_children.items():
                # この内部ノードが含むクラスタを取得
                contained_clusters = get_all_clusters(parent_node_id)

                # 複数のクラスタを含む場合のみ親として扱う
                if len(contained_clusters) > 1:
                    # 実際のクラスタノードを取得
                    child_nodes = [cluster_id_to_node[c] for c in contained_clusters if c in cluster_id_to_node]

                    if len(child_nodes) > 1:
                        # 親ノードの代表語は子の中で最初の用語
                        parent_representative = child_nodes[0].representative

                        # 親クラスタノードを作成
                        parent_node = SynonymHierarchy(
                            representative=f"親:{parent_representative}",
                            terms=[],  # 親ノードは直接用語を持たない
                            cluster_id=parent_node_id,
                            level=1,  # 親レベル
                            category_name=f"上位概念グループ"
                        )

                        # 子ノードを親に追加
                        for child_node in child_nodes:
                            parent_node.children[child_node.representative] = child_node
                            child_node.level = 0  # 子レベル

                            # 子を hierarchies から削除（親の下に移動）
                            if child_node.representative in hierarchies:
                                del hierarchies[child_node.representative]

                        # 親をhierarchiesに追加
                        hierarchies[parent_node.representative] = parent_node
                        created_parents.add(parent_node_id)

            import logging
            logger = logging.getLogger(__name__)
            if created_parents:
                logger.info(f"階層構造: {len(created_parents)}個の親クラスタを生成")
            else:
                logger.info(f"階層構造: 親クラスタなし（全て葉ノード）")

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"condensed_treeからの階層抽出失敗: {e}", exc_info=True)
            # エラー時はフラット構造のまま

    def _select_representative(self, terms: List[Term]) -> str:
        """クラスタの代表語を選定（最短の用語）"""
        return min(terms, key=lambda t: len(t.term)).term

    def _generate_category_names(
        self,
        hierarchy: Dict[str, SynonymHierarchy],
        all_terms: List[Term],
        verbose: bool = True
    ):
        """各クラスタにLLMでカテゴリ名を付与"""
        term_dict = {t.term: t for t in all_terms}

        prompt = ChatPromptTemplate.from_messages(
            get_category_naming_prompt_messages()
        )
        chain = prompt | self.llm | StrOutputParser()

        for i, (rep, node) in enumerate(hierarchy.items(), 1):
            if verbose and i % 5 == 0:
                print(f"  {i}/{len(hierarchy)}件処理中...")

            terms_with_defs = []
            for term_name in node.terms:
                if term_name in term_dict:
                    term = term_dict[term_name]
                    terms_with_defs.append(
                        f"用語: {term.term}\n定義: {term.definition[:200]}"
                    )

            terms_text = "\n\n".join(terms_with_defs)

            result_text = chain.invoke({
                "terms_with_definitions": terms_text
            })

            result = self._parse_llm_result(result_text)

            if result:
                node.category_name = result.get("category", "")
                node.category_confidence = result.get("confidence", 0.0)
                node.category_reason = result.get("reason", "")

    def _parse_llm_result(self, text: str) -> Optional[Dict]:
        """LLM結果のJSONをパース"""
        text = text.strip()

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


def extract_synonym_hierarchy(
    terms: List[Term],
    config: Optional[Config] = None,
    min_cluster_size: int = 2,
    generate_category_names: bool = True,
    use_umap: bool = False,
    umap_n_components: int = 50,
    verbose: bool = True
) -> Dict[str, SynonymHierarchy]:
    """
    階層的類義語構造を抽出（簡易関数）

    Args:
        terms: 定義付き用語リスト
        config: 設定
        min_cluster_size: 最小クラスタサイズ
        generate_category_names: LLMでカテゴリ名を生成するか
        use_umap: UMAP次元削減を使用するか
        umap_n_components: UMAP削減後の次元数
        verbose: 進捗表示

    Returns:
        {代表語: SynonymHierarchy} の辞書
    """
    extractor = HierarchicalSynonymExtractor(
        config=config,
        min_cluster_size=min_cluster_size,
        use_umap=use_umap,
        umap_n_components=umap_n_components
    )

    return extractor.extract_hierarchy(
        terms,
        generate_category_names=generate_category_names,
        verbose=verbose
    )

def generate_cluster_category_names(
    hierarchy: Dict[str, SynonymHierarchy],
    terms: List[Term],
    config: Optional[Config] = None,
    llm_model: str = "gpt-4.1-mini",
    verbose: bool = True
) -> Dict[str, SynonymHierarchy]:
    """
    クラスタにカテゴリ名を生成（スタンドアロン関数）
    
    Args:
        hierarchy: クラスタ階層情報
        terms: 専門用語リスト（定義付き）
        config: 設定
        llm_model: LLMモデル名
        verbose: 進捗表示
        
    Returns:
        カテゴリ名が付与された階層情報
    """
    if not hierarchy:
        return hierarchy
        
    config = config or Config()
    
    # LLM初期化
    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from dictionary_system.config.prompts import get_category_naming_prompt_messages
    import json
    
    llm = AzureChatOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        azure_deployment=llm_model,
        temperature=0.1,
        max_tokens=500
    )
    
    term_dict = {t.term: t for t in terms}
    
    prompt = ChatPromptTemplate.from_messages(
        get_category_naming_prompt_messages()
    )
    chain = prompt | llm | StrOutputParser()
    
    if verbose:
        print(f"カテゴリ名生成: {len(hierarchy)}クラスタ")
    
    for i, (rep, node) in enumerate(hierarchy.items(), 1):
        if verbose and i % 5 == 0:
            print(f"  {i}/{len(hierarchy)}件処理中...")
        
        # このクラスタの用語と定義を取得
        terms_with_defs = []
        for term_name in node.terms:
            if term_name in term_dict:
                term = term_dict[term_name]
                terms_with_defs.append(
                    f"用語: {term.term}\n定義: {(term.definition or '')[:200]}"
                )
        
        if not terms_with_defs:
            continue
            
        terms_text = "\n\n".join(terms_with_defs)
        
        try:
            result_text = chain.invoke({
                "terms_with_definitions": terms_text
            })
            
            # JSON解析
            result_text = result_text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            if result:
                node.category_name = result.get("category", "")
                node.category_confidence = result.get("confidence", 0.0)
                node.category_reason = result.get("reason", "")
        except Exception as e:
            if verbose:
                print(f"  カテゴリ名生成エラー: {e}")
            node.category_name = f"クラスタ{node.cluster_id}"
            node.category_confidence = 0.0
    
    return hierarchy
