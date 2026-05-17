from sqlalchemy import create_engine
import os 
from dotenv import load_dotenv
import pandas as pd 
import streamlit as st 
# Test for update

class DB:
    if os.getenv("DB_USERNAME") is None or os.getenv("DB_PASSWORD") is None:
        load_dotenv()

    def __init__(self):
        self.username = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.database = 'LENDBI_DATAMART'
        
    def connection(self) -> None:
        connection_url = f"mssql+pyodbc://{self.username}:{self.password}@LEN-RBBIREPORT/{self.database}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(connection_url)
        self.conn = engine.connect()

    def query_from_file(self, path:str) -> pd.DataFrame:
        try:
            with open(path, encoding='utf-8') as file:
                query = file.read()
        except EncodingWarning as e:
            st.error(f'The sql file specified is not encoded in utf-8, please encode correctly: {e}')
        return pd.read_sql_query(query, con=self.conn)

    def query_str(self, query:str) -> pd.DataFrame:
        return pd.read_sql_query(query,self.conn)
    
    def close_connection(self):
        self.conn.close()

        
