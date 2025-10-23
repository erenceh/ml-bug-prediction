import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")

def get_connection():
  conn = sqlite3.connect(DB_PATH, check_same_thread=False)
  conn.row_factory = sqlite3.Row
  return conn

def init_db():
  conn = get_connection()
  conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT,
      description TEXT,
      predicted_label TEXT,
      numeric_label INTEGER,
      confidence REAL,
      timestamp TEXT
    )
  """)
  conn.commit()
  conn.close()