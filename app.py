#!/usr/bin/env python3
"""
専門用語抽出システム - Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import tempfile
import sys
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from dictionary_system.core.extractors.statistical_extractor_v2 import EnhancedTermExtractorV3
import asyncio
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
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# タイトルとヘッダー
st.title("📚 専門用語辞書自動構築システム")
st.markdown("PDFドキュメントから専門用語を自動抽出し、重要度スコアを計算します。")

# サイドバー - パラメータ設定
with st.sidebar:
    st.header("⚙️ パラメータ設定")

    # Azure OpenAI設定
    st.subheader("🤖 Azure OpenAI設定")
    azure_endpoint = st.text_input(
        "Azure Endpoint",
        value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        type="password",
        help="Azure OpenAIのエンドポイントURL"
    )
    azure_api_key = st.text_input(
        "API Key",
        value=os.getenv("AZURE_OPENAI_API_KEY", ""),
        type="password",
        help="Azure OpenAIのAPIキー"
    )

    st.subheader("基本設定")
    min_frequency = st.slider(
        "最小出現頻度",
        min_value=1,
        max_value=10,
        value=2,
        help="1に設定すると見出しの用語も抽出します"
    )

    col1, col2 = st.columns(2)
    with col1:
        min_term_length = st.number_input(
            "最小文字数",
            min_value=1,
            max_value=10,
            value=2
        )
    with col2:
        max_term_length = st.number_input(
            "最大文字数",
            min_value=5,
            max_value=30,
            value=15
        )

    st.subheader("詳細設定")
    k_neighbors = st.slider(
        "kNN近傍数",
        min_value=5,
        max_value=20,
        value=10,
        help="グラフ構築時の近傍ノード数"
    )

    sim_threshold = st.slider(
        "類似度閾値",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help="エッジを作成する最小類似度"
    )

    top_n = st.number_input(
        "表示件数",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help="結果表示する上位件数"
    )

    use_cache = st.checkbox(
        "埋め込みキャッシュを使用",
        value=True,
        help="計算済み埋め込みベクトルを再利用"
    )

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
        st.success(f"✅ ファイルがアップロードされました: {uploaded_file.name}")
        st.session_state.uploaded_file_name = uploaded_file.name

        # ファイル情報表示
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

            # プログレスバー表示
            progress_bar = st.progress(0, text="抽出処理を開始しています...")

            try:
                # Azure OpenAI設定を環境変数に設定
                if azure_endpoint and azure_api_key:
                    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
                else:
                    st.error("⚠️ Azure OpenAIの設定が必要です")
                    st.stop()

                # 抽出器の初期化
                progress_bar.progress(10, text="抽出器を初期化中...")
                extractor = EnhancedTermExtractorV3(
                    min_frequency=min_frequency,
                    min_term_length=min_term_length,
                    max_term_length=max_term_length,
                    k_neighbors=k_neighbors,
                    sim_threshold=sim_threshold,
                    use_cache=use_cache,
                    use_llm_validation=True,
                    use_azure_openai=True,
                    use_rag_context=True
                )

                # 抽出実行（非同期関数を実行）
                progress_bar.progress(30, text="PDFからテキストを抽出中...")
                with st.spinner("専門用語を抽出しています（LLM検証を含む）..."):
                    # 非同期関数を同期的に実行
                    terms = asyncio.run(extractor.extract_terms_with_validation(tmp_path))

                progress_bar.progress(100, text="抽出完了！")

                # 結果を保存
                st.session_state.extraction_results = terms

                # デバッグ: 最初の3件の値を表示
                if terms and len(terms) > 0:
                    st.write("[DEBUG] 抽出結果サンプル（最初の3件）:")
                    for i, term in enumerate(terms[:3]):
                        st.write(f"  {i+1}. {term['term']}: TF-IDF={term.get('tfidf', 'N/A')}, C-value={term.get('c_value', 'N/A')}")

                st.success(f"✨ {len(terms)}個の専門用語を抽出しました！")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
            finally:
                # 一時ファイルを削除
                Path(tmp_path).unlink(missing_ok=True)
                progress_bar.empty()
    else:
        st.info("👈 まずPDFファイルをアップロードしてください")

# 結果表示エリア
if st.session_state.extraction_results:
    st.header("📊 抽出結果")

    results = st.session_state.extraction_results[:top_n]

    # タブで表示を切り替え
    tab1, tab2, tab3 = st.tabs(["📋 抽出結果一覧", "📈 視覚化", "💾 エクスポート"])

    with tab1:
        # データフレーム作成
        df = pd.DataFrame(results)
        df.index = range(1, len(df) + 1)
        df.index.name = "順位"

        # 列名を日本語に変換
        column_mapping = {
            'term': '専門用語',
            'score': '総合スコア',
            'frequency': '出現頻度',
            'c_value': 'C-value',
            'tfidf': 'TF-IDF',
            'pagerank': 'PageRank'
        }

        # 存在する列のみマッピング
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # スコアを小数点3桁に整形
        for col in ['総合スコア', 'C-value', 'TF-IDF', 'PageRank']:
            if col in df.columns:
                df[col] = df[col].round(3)

        # 表示
        st.dataframe(
            df,
            height=500,
            column_config={
                "専門用語": st.column_config.TextColumn(
                    "専門用語",
                    width="medium"
                ),
                "総合スコア": st.column_config.ProgressColumn(
                    "総合スコア",
                    min_value=0,
                    max_value=1,
                    format="%.3f"
                ),
                "出現頻度": st.column_config.NumberColumn(
                    "出現頻度",
                    format="%d"
                )
            }
        )

        # 統計情報
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総抽出数", len(st.session_state.extraction_results))
        with col2:
            avg_score = np.mean([t['score'] for t in results])
            st.metric("平均スコア", f"{avg_score:.3f}")
        with col3:
            total_freq = sum([t['frequency'] for t in results])
            st.metric("総出現回数", total_freq)
        with col4:
            unique_freq_1 = len([t for t in results if t['frequency'] == 1])
            st.metric("頻度1の用語", unique_freq_1)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # スコア分布
            st.subheader("スコア分布")
            fig_score = px.histogram(
                df,
                x='総合スコア',
                nbins=20,
                title="総合スコアの分布",
                labels={'count': '用語数'}
            )
            st.plotly_chart(fig_score, use_responsive_container_width=True)

        with col2:
            # 頻度分布（対数スケール）
            st.subheader("出現頻度分布")
            freq_counts = df['出現頻度'].value_counts().sort_index()
            fig_freq = px.bar(
                x=freq_counts.index,
                y=freq_counts.values,
                title="出現頻度ごとの用語数",
                labels={'x': '出現頻度', 'y': '用語数'},
                log_y=True
            )
            st.plotly_chart(fig_freq, use_responsive_container_width=True)

        # 散布図
        st.subheader("スコア相関")
        if 'C-value' in df.columns and 'TF-IDF' in df.columns:
            fig_scatter = px.scatter(
                df,
                x='C-value',
                y='TF-IDF',
                size='出現頻度',
                color='総合スコア',
                hover_data=['専門用語'],
                title="C-value vs TF-IDF",
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_scatter, use_responsive_container_width=True)

    with tab3:
        st.subheader("データエクスポート")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV形式
            csv = df.to_csv(index=True, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSVダウンロード",
                data=csv,
                file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # JSON形式
            import json
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSONダウンロード",
                data=json_str,
                file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col3:
            # 上位用語のみテキスト形式
            text_output = "\n".join([f"{i+1}. {t['term']} ({t['score']:.3f})"
                                    for i, t in enumerate(results[:20])])
            st.download_button(
                label="📥 上位20語（テキスト）",
                data=text_output,
                file_name=f"top_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        # プレビュー
        with st.expander("エクスポートデータのプレビュー"):
            st.code(text_output, language=None)

# フッター
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    専門用語辞書自動構築システム v1.0 |
    <a href='https://github.com/uchi736/Dictionary-of-Technical-Terms' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)