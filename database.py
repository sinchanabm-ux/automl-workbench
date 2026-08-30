import sqlite3

# connecting database
conn = sqlite3.connect('runs.db')

cursor = conn.cursor()

#creating table
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
