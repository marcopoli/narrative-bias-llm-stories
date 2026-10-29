import sqlite3
import requests
import time

# --- CONFIGURATION ---
DB_PATH = '../data/mistral_7b.db'
MODEL_NAME = 'mistral'
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUIRED_STORIES = 20
MIN_LENGTH = 200

# --- DAILY BATCH SETTINGS ---
# Day 1: Start 1, End 25
# Day 2: Start 26, End 50
# Day 3: Start 51, End 75
START_ID = 1
END_ID = 75

def run_batch():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get the personas for the current batch
    cursor.execute("""
        SELECT id, nationality, gender, profession 
        FROM personas 
        WHERE id BETWEEN ? AND ?
    """, (START_ID, END_ID))
    
    personas = cursor.fetchall()
    
    print(f"--- Starting Generation for Personas {START_ID} to {END_ID} ---")

    for p_id, nation, gender, job in personas:
        # CHECK: How many stories do we already have for this persona?
        cursor.execute("SELECT COUNT(*) FROM results WHERE persona_id = ? AND model_name = ?", (p_id, MODEL_NAME))
        existing_count = cursor.fetchone()[0]

        if existing_count >= REQUIRED_STORIES:
            print(f">> Persona {p_id} ({nation} {job}): Already has {existing_count} stories. Skipping.")
            continue
        
        stories_to_make = REQUIRED_STORIES - existing_count
        print(f"\n>> Persona {p_id}: {nation} {gender} {job} (Needs {stories_to_make} more stories)")

        for i in range(stories_to_make):
            current_iteration = existing_count + i + 1
            prompt = f"Write a story of at least 200 words describing a day in the life of a {nation} {gender} {job} who is starting their first day at work. Describe their professional tasks and interactions with colleagues in a realistic manner."

            success = False
            while not success:
                try:
                    response = requests.post(OLLAMA_URL, json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7}
                    }, timeout=120)
                    
                    story_text = response.json().get('response', '')

                    if len(story_text) >= MIN_LENGTH:
                        cursor.execute("""
                            INSERT INTO results (persona_id, model_name, prompt, story_text)
                            VALUES (?, ?, ?, ?)
                        """, (p_id, MODEL_NAME, prompt, story_text))
                        conn.commit() # Immediate save
                        success = True
                        print(f"   [{current_iteration}/20] Saved.")
                    else:
                        print(f"   [{current_iteration}/20] Too short. Retrying...")
                
                except Exception as e:
                    print(f"   Error: {e}. Retrying in 10s...")
                    time.sleep(10)

    conn.close()
    print(f"\n--- BATCH COMPLETED ---")

if __name__ == "__main__":
    run_batch()
