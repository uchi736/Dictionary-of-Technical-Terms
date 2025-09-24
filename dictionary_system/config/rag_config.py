#!/usr/bin/env python3
"""
設定管理クラス
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """環境設定管理クラス"""

    def __init__(self, env_file: Optional[str] = None):
        """
        初期化

        Args:
            env_file: .envファイルパス（指定しない場合はデフォルト）
        """
        if env_file:
            load_dotenv(env_file)
        else:
            # プロジェクトルートの.envを探す
            current_dir = Path(__file__).parent.parent.parent
            env_path = current_dir / '.env'
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()  # システムの環境変数を使用

        # Azure OpenAI設定
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY', '')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
        self.azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        self.azure_openai_chat_deployment_name = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME', 'gpt-4')
        self.azure_openai_embedding_deployment_name = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-ada-002')

        # OpenAI API設定
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')

        # PostgreSQL設定
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'dictionary_db')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', 'postgres')

        # RAG設定
        self.jargon_table_name = os.getenv('JARGON_TABLE_NAME', 'technical_terms')
        self.vector_dimension = int(os.getenv('VECTOR_DIMENSION', '1536'))

        # LangSmith設定
        self.langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'
        self.langchain_api_key = os.getenv('LANGCHAIN_API_KEY', '')
        self.langchain_project = os.getenv('LANGCHAIN_PROJECT', 'term_extraction')

        # ログ設定
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'extraction.log')

        # キャッシュ設定
        self.use_cache = os.getenv('USE_CACHE', 'true').lower() == 'true'
        self.cache_dir = os.getenv('CACHE_DIR', 'cache')

        # 抽出パラメータ
        self.min_term_length = int(os.getenv('MIN_TERM_LENGTH', '2'))
        self.max_term_length = int(os.getenv('MAX_TERM_LENGTH', '20'))
        self.min_frequency = int(os.getenv('MIN_FREQUENCY', '2'))

        # モデル設定
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'

    def get_db_url(self) -> str:
        """PostgreSQL接続URLを取得"""
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def is_azure_configured(self) -> bool:
        """Azure OpenAIが設定されているか確認"""
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    def is_openai_configured(self) -> bool:
        """OpenAI APIが設定されているか確認"""
        return bool(self.openai_api_key)

    def is_db_configured(self) -> bool:
        """データベースが設定されているか確認"""
        return bool(self.db_host and self.db_name)

    def validate(self) -> bool:
        """設定の妥当性を検証"""
        # 少なくとも1つのAI APIが設定されているか
        if not (self.is_azure_configured() or self.is_openai_configured()):
            print("Warning: No AI API configured (Azure OpenAI or OpenAI)")

        # データベース設定の確認
        if not self.is_db_configured():
            print("Warning: Database not configured")

        return True

    def print_config(self):
        """設定内容を表示（パスワードは隠す）"""
        print("=== Configuration ===")
        print(f"Azure OpenAI configured: {self.is_azure_configured()}")
        print(f"OpenAI API configured: {self.is_openai_configured()}")
        print(f"Database configured: {self.is_db_configured()}")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Cache enabled: {self.use_cache}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Min term length: {self.min_term_length}")
        print(f"Max term length: {self.max_term_length}")
        print(f"Min frequency: {self.min_frequency}")
        print("====================")