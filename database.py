import sqlite3
from datetime import datetime

def create_table():
    conn = sqlite3.connect('runs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs(
            id  INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            target_column TEXT,
            problem_type TEXT,
            model_name TEXT,
            score REAL,
            timestamp TEXT    
        )
    ''')
    conn.commit()
    conn.close()

def log_run(filename, target_column, problem_type, model_name, score):
    conn = sqlite3.connect('runs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO runs (filename, target_column, problem_type, model_name, score, timestamp) VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, target_column, problem_type, model_name, score, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
