#!/usr/bin/env python3
"""
専門用語抽出システム - Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import sys
from datetime import datetime
import fitz

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from dictionary_system.core.extractors.enhanced_term_extractor_v4 import EnhancedTermExtractorV4
import os
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="専門用語抽出システム",
    page_icon="📚",
    layout="wide"
)

# セッション状態の初期化
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None
if 'hierarchy' not in st.session_state:
    st.session_state.hierarchy = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

st.title("📚 専門用語辞書自動構築システム V4 (TF-IDF + C-value + SemRe-Rank + RAG)")
st.markdown("PDFから専門用語を抽出 → 定義生成 → LLM判定 → 階層的類義語グループ化")

with st.sidebar:
    st.header("⚙️ パラメータ設定")

    st.subheader("🔤 基本設定")
    min_term_length = st.slider("最小用語長", 2, 5, 2)
    max_term_length = st.slider("最大用語長", 5, 15, 8)
    min_frequency = st.slider("最小出現回数", 1, 5, 2)

    st.subheader("🎯 SemRe-Rank設定")
    seed_z = st.slider(
        "シード選定上限",
        min_value=5,
        max_value=20,
        value=10,
        help="上位何件からシードを選定するか（エルボー法無効、固定数選定）"
    )

    min_seed_count = st.slider("最小シード数", 3, 20, 5)
    max_seed_ratio = st.slider("最大シード比率", 0.1, 0.5, 0.2, 0.05)

    relmin = st.slider("relmin (最小類似度)", 0.0, 1.0, 0.5, 0.1)
    reltop = st.slider("reltop (上位割合)", 0.05, 0.5, 0.15, 0.05)

    st.subheader("📖 定義生成 (RAG)")
    enable_definition = st.checkbox("定義生成を有効化", value=True)
    if enable_definition:
        col1, col2, col3 = st.columns(3)
        with col1:
            definition_percentage = st.slider(
                "定義生成割合(%)",
                10, 50, 25, 5,
                help="候補の上位何%に定義を生成するか"
            )
        with col2:
            min_definitions = st.number_input(
                "最小定義生成数",
                min_value=10, max_value=30, value=15, step=5,
                help="文書が短い場合でも最低限生成する数"
            )
        with col3:
            max_definitions = st.number_input(
                "最大定義生成数",
                min_value=30, max_value=100, value=50, step=10,
                help="処理時間制御のための上限"
            )
        # V4では内部で動的計算するので、この値は使わない
        top_n_definition = max_definitions
    else:
        top_n_definition = None

    st.subheader("🔍 LLM専門用語判定")
    enable_filtering = st.checkbox("専門用語フィルタリングを有効化", value=True)

    st.subheader("🌳 階層的類義語抽出")
    enable_hierarchy = st.checkbox("階層的類義語抽出を有効化", value=True)
    min_cluster_size = st.slider("最小クラスタサイズ", 2, 5, 2) if enable_hierarchy else 2
    generate_category_names = st.checkbox("LLMでカテゴリ名生成", value=True) if enable_hierarchy else False

    use_umap = st.checkbox("UMAP次元削減を使用", value=False, help="高次元埋め込みを低次元に圧縮（クラスタリング精度向上）") if enable_hierarchy else False
    umap_n_components = st.slider("UMAP削減次元数", 30, 100, 50, 10, help="1536次元からの削減先") if (enable_hierarchy and use_umap) else 50

# メインエリア
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 PDFアップロード")

    uploaded_file = st.file_uploader(
        "PDFファイルを選択してください",
        type=['pdf'],
        help="専門用語を抽出したいPDFファイルをアップロード"
    )

    if uploaded_file is not None:
        st.success(f"✅ ファイル: {uploaded_file.name}")
        st.session_state.uploaded_file_name = uploaded_file.name

        file_details = {
            "ファイル名": uploaded_file.name,
            "ファイルサイズ": f"{uploaded_file.size / 1024:.1f} KB",
            "ファイルタイプ": uploaded_file.type
        }
        for key, value in file_details.items():
            st.text(f"{key}: {value}")

with col2:
    st.header("🚀 抽出実行")

    if uploaded_file is not None:
        if st.button("専門用語を抽出", type="primary"):

            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            progress_bar = st.progress(0, text="処理開始...")

            try:
                progress_bar.progress(10, text="PDFからテキスト抽出中...")
                doc = fitz.open(tmp_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()

                st.info(f"抽出テキスト: {len(text)}文字")

                progress_bar.progress(20, text="専門用語抽出開始 (V4: TF-IDF + C-value + SemRe-Rank)...")

                extractor = EnhancedTermExtractorV4(
                    min_term_length=min_term_length,
                    max_term_length=max_term_length,
                    min_frequency=min_frequency,
                    use_azure_openai=True,
                    seed_z=seed_z,
                    use_elbow_detection=False,
                    min_seed_count=min_seed_count,
                    max_seed_ratio=max_seed_ratio,
                    relmin=relmin,
                    reltop=reltop,
                    enable_definition_generation=enable_definition,
                    enable_definition_filtering=enable_filtering,
                    enable_synonym_hierarchy=enable_hierarchy,
                    top_n_definition=top_n_definition,
                    min_cluster_size=min_cluster_size,
                    generate_category_names=generate_category_names,
                    use_umap=use_umap,
                    umap_n_components=umap_n_components
                )

                progress_bar.progress(40, text="用語抽出中...")
                terms = extractor.extract(text)

                st.session_state.extraction_results = terms

                if enable_definition:
                    progress_bar.progress(60, text="定義生成中 (RAG)...")
                    def_count = len([t for t in terms if t.definition])
                    st.success(f"✅ {def_count}件に定義生成")

                if enable_filtering:
                    progress_bar.progress(75, text="LLM専門用語判定中...")

                if enable_hierarchy:
                    progress_bar.progress(85, text="階層的類義語抽出中 (HDBSCAN)...")
                    if hasattr(extractor, 'hierarchy') and extractor.hierarchy:
                        st.session_state.hierarchy = extractor.hierarchy
                        st.success(f"✅ {len(extractor.hierarchy)}個のクラスタを生成")

                progress_bar.progress(100, text="完了!")
                st.success(f"✅ {len(terms)}件の専門用語を抽出")

            except Exception as e:
                st.error(f"エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                progress_bar.empty()
    else:
        st.info("👈 まずPDFファイルをアップロードしてください")

# 結果表示エリア
if st.session_state.extraction_results:
    st.header("📊 抽出結果")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 専門用語一覧",
        "📖 定義付き用語",
        "🌳 階層的クラスタ",
        "💾 エクスポート"
    ])

    with tab1:
        st.subheader("抽出された専門用語 (Top 50)")
        results = st.session_state.extraction_results[:50]

        df = pd.DataFrame([{
            '順位': i+1,
            '専門用語': t.term,
            'スコア': round(t.score, 4)
        } for i, t in enumerate(results)])

        st.dataframe(
            df,
            hide_index=True,
            height=500
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("総抽出数", len(st.session_state.extraction_results))
        with col2:
            avg_score = np.mean([t.score for t in results])
            st.metric("平均スコア", f"{avg_score:.3f}")

    with tab2:
        enriched = [t for t in st.session_state.extraction_results if t.definition]

        if enriched:
            st.subheader("定義付き専門用語")

            for i, term in enumerate(enriched[:20], 1):
                with st.expander(f"{i}. **{term.term}** (スコア: {term.score:.3f})"):
                    st.markdown(f"**定義:**\n\n{term.definition}")

            st.metric("定義付き用語数", len(enriched))
        else:
            st.info("定義はまだ生成されていません")

    with tab3:
        if st.session_state.hierarchy:
            st.subheader("階層的類義語クラスタ")

            hierarchy = st.session_state.hierarchy

            # クラスタ統計
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("クラスタ数", len(hierarchy))
            with col2:
                clustered = sum(len(node.terms) for node in hierarchy.values())
                st.metric("クラスタ化された用語", clustered)
            with col3:
                enriched = [t for t in st.session_state.extraction_results if t.definition]
                if enriched:
                    noise = len(enriched) - clustered
                    st.metric("ノイズ (未クラスタ化)", noise)

            # 表示モード選択
            view_mode = st.radio(
                "表示モード",
                ["ツリービュー", "リストビュー"],
                horizontal=True
            )

            if view_mode == "ツリービュー":
                # ツリー形式表示
                st.markdown("### 📁 階層ツリー")

                # 用語→クラスタのマッピングを作成
                term_to_cluster = {}
                for rep, node in hierarchy.items():
                    for term in node.terms:
                        term_to_cluster[term] = (node.category_name or rep, node)

                # カテゴリ別にソート
                sorted_hierarchy = sorted(
                    hierarchy.items(),
                    key=lambda x: x[1].category_name or x[0]
                )

                for i, (rep, node) in enumerate(sorted_hierarchy, 1):
                    category = node.category_name or f"クラスタ {i}"

                    # カテゴリレベル
                    with st.expander(f"📂 **{category}** ({len(node.terms)}件)", expanded=False):
                        if node.category_reason:
                            st.caption(f"💡 {node.category_reason}")

                        # 用語リスト（ツリー形式）
                        for term in sorted(node.terms):
                            # 用語の詳細情報を取得
                            term_obj = next((t for t in st.session_state.extraction_results if t.term == term), None)

                            if term_obj:
                                score_str = f"(スコア: {term_obj.score:.3f})" if hasattr(term_obj, 'score') else ""
                                st.markdown(f"└─ **{term}** {score_str}")

                                if term_obj.definition:
                                    with st.container():
                                        st.caption(f"📝 {term_obj.definition[:150]}..." if len(term_obj.definition) > 150 else f"📝 {term_obj.definition}")
                            else:
                                st.markdown(f"└─ {term}")

            else:
                # 既存のリストビュー
                for i, (rep, node) in enumerate(hierarchy.items(), 1):
                    with st.expander(
                        f"クラスタ {i}: **{node.category_name or rep}** "
                        f"({len(node.terms)}件)"
                    ):
                        if node.category_name:
                            st.markdown(f"**カテゴリ:** {node.category_name}")
                            st.markdown(f"**信頼度:** {node.category_confidence:.2f}")
                            if node.category_reason:
                                st.caption(node.category_reason)

                        st.markdown("**含まれる用語:**")
                        terms_list = ", ".join(node.terms)
                        st.markdown(f"_{terms_list}_")

            # ネットワークグラフ可視化
            st.subheader("クラスタネットワーク")

            # クラスタ間の関係を可視化
            node_x = []
            node_y = []
            node_text = []
            node_size = []

            for i, (rep, node) in enumerate(hierarchy.items()):
                angle = 2 * np.pi * i / len(hierarchy)
                node_x.append(np.cos(angle))
                node_y.append(np.sin(angle))
                node_text.append(
                    f"{node.category_name or rep}<br>{len(node.terms)}件"
                )
                node_size.append(len(node.terms) * 10 + 20)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(size=node_size, color='lightblue'),
                text=node_text,
                textposition='top center',
                hoverinfo='text'
            ))

            fig.update_layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("階層的クラスタはまだ生成されていません")

    with tab4:
        st.subheader("データエクスポート")

        col1, col2 = st.columns(2)

        with col1:
            # 専門用語リスト CSV
            if st.session_state.extraction_results:
                df_export = pd.DataFrame([{
                    '専門用語': t.term,
                    'スコア': t.score
                } for t in st.session_state.extraction_results[:50]])

                csv = df_export.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 専門用語リスト (CSV)",
                    data=csv,
                    file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            # 階層的クラスタ JSON
            if st.session_state.hierarchy:
                import json
                import numpy as np
                hierarchy_dict = {}
                for rep, node in st.session_state.hierarchy.items():
                    hierarchy_dict[rep] = {
                        'category': node.category_name,
                        'confidence': float(node.category_confidence) if isinstance(node.category_confidence, (np.floating, np.integer)) else node.category_confidence,
                        'terms': node.terms,
                        'cluster_id': int(node.cluster_id) if isinstance(node.cluster_id, (np.integer, np.int64)) else node.cluster_id
                    }

                json_str = json.dumps(hierarchy_dict, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 階層的クラスタ (JSON)",
                    data=json_str,
                    file_name=f"hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# フッター
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    専門用語辞書自動構築システム v4.0 (TF-IDF + C-value + SemRe-Rank + RAG + HDBSCAN) |
    <a href='https://github.com/uchi736/Dictionary-of-Technical-Terms' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)