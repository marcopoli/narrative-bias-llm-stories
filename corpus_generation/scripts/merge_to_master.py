import sqlite3
import os

# --- CONFIGURATION ---
SOURCE_FILES = [
    '../data/gemma3_4b.db',
    '../data/mistral_7b.db',
    '../data/llama3_8b.db'
]
MASTER_DB = '../../analysis/master_results.db'

def merge_databases():
    print("--- Starting Master Merge Process ---")
    
    # 1. Start with a fresh Master DB
    if os.path.exists(MASTER_DB):
        os.remove(MASTER_DB)
        print(f"Removed existing {MASTER_DB} to ensure a clean start.")

    master_conn = sqlite3.connect(MASTER_DB)
    master_cursor = master_conn.cursor()

    # 2. Create the Final Schema (All columns included)
    print("Creating Master Schema...")
    master_cursor.execute('''
        CREATE TABLE personas (
            id INTEGER PRIMARY KEY,
            nationality TEXT,
            gender TEXT,
            profession TEXT
        )
    ''')

    master_cursor.execute('''
        CREATE TABLE results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id INTEGER,
            model_name TEXT,
            prompt TEXT,
            story_text TEXT,
            sentiment_score REAL,
            sentiment_volatility REAL,
            agency_start REAL,
            agency_middle REAL,
            agency_end REAL,
            agency_arc REAL,
            agency_quality_score REAL,
            concreteness_score REAL,
            taxonomy_label TEXT,
            FOREIGN KEY (persona_id) REFERENCES personas(id)
        )
    ''')

    # 3. Import the 75 Personas (using the Gemma file as the reference)
    print("Importing 75 persona definitions...")
    ref_conn = sqlite3.connect(SOURCE_FILES[0])
    personas = ref_conn.execute("SELECT id, nationality, gender, profession FROM personas").fetchall()
    master_cursor.executemany("INSERT INTO personas VALUES (?,?,?,?)", personas)
    ref_conn.close()

    # 4. Merge all Stories from the 3 Source Files
    for file_path in SOURCE_FILES:
        if not os.path.exists(file_path):
            print(f"CRITICAL ERROR: {file_path} not found! Check your folder.")
            continue
            
        print(f"Reading stories from {file_path}...")
        source_conn = sqlite3.connect(file_path)
        source_cursor = source_conn.cursor()
        
        # We select only the content columns. 
        # The 'id' is skipped because the master DB will create its own unique IDs (1 to 4500).
        source_cursor.execute("""
            SELECT persona_id, model_name, prompt, story_text 
            FROM results
        """)
        
        stories = source_cursor.fetchall()
        
        # Insert into master (analysis columns will be NULL for now)
        master_cursor.executemany("""
            INSERT INTO results (persona_id, model_name, prompt, story_text)
            VALUES (?, ?, ?, ?)
        """, stories)
        
        print(f"   > Successfully imported {len(stories)} stories.")
        source_conn.close()

    master_conn.commit()
    
    # 5. Final Check
    total_records = master_cursor.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    print(f"\n{'='*40}")
    print(f"MERGE SUCCESSFUL!")
    print(f"Total Stories in Master DB: {total_records}")
    print(f"Master file saved at: {MASTER_DB}")
    print(f"{'='*40}")
    
    master_conn.close()

if __name__ == "__main__":
    merge_databases()