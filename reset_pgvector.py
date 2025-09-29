#!/usr/bin/env python3
"""Reset PGVector tables"""

from dictionary_system.config.rag_config import Config
from sqlalchemy import create_engine, text

config = Config()
engine = create_engine(config.get_db_url())

with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS langchain_pg_collection CASCADE'))
    conn.execute(text('DROP TABLE IF EXISTS langchain_pg_embedding CASCADE'))
    conn.commit()

print('PGVector tables dropped successfully')