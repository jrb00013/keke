import sqlite3
import os

DB_FILE = "database.db"

def initialize_db():
    if os.path.exists(DB_FILE):
        print("📌 Database already exists. Skipping initialization.")
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create stock market data table
    cursor.execute('''
        CREATE TABLE stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    # Create AI prediction results table
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    conn.commit()
    conn.close()
    print("✅ Database Initialized Successfully!")

if __name__ == "__main__":
    initialize
