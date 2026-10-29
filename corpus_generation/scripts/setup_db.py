"""
setup_db.py — Database Initialisation for Narrative Bias in LLMs

Creates the SQLite schema for the full study pipeline and populates
the 75 persona combinations (5 nationalities × 3 genders × 5 professions).

Run this once before any generator script:
    python setup_db.py

Tables created:
    personas   — 75 rows, one per unique identity combination
    results    — 4,500 rows after generation (75 personas × 3 models × 20 stories)

4-Pillar Analysis Schema (columns added by master_analysis.py):
    P1  Atmospheric Sentiment    → sentiment_score, sentiment_volatility
    P2  Three-Act Agency Arc     → agency_start, agency_middle, agency_end, agency_arc
    P3  Agency Quality           → agency_quality_score
    P4  Semantic Concreteness    → concreteness_score
    P5  Thematic Taxonomy (BART) → taxonomy_label  [negative result — see paper]
"""

import sqlite3
import os

# ── Path Setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
DB_PATH  = os.path.join(BASE_DIR, '..', '..', 'analysis', 'master_results.db')

os.makedirs(DATA_DIR, exist_ok=True)

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print(f"Connected to: {DB_PATH}")

# ── Table 1: Personas ──────────────────────────────────────────────────────────
# 75 unique identity combinations across nationality × gender × profession.
# IDs are deterministic (insertion order) — all three generators must share
# this exact ordering so persona_id is consistent across per-model databases.
cursor.execute('''
    CREATE TABLE IF NOT EXISTS personas (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        nationality TEXT NOT NULL,
        gender      TEXT NOT NULL,
        profession  TEXT NOT NULL
    )
''')

# ── Table 2: Results ───────────────────────────────────────────────────────────
# One row per generated story. Story text and prompt are stored verbatim for
# full reproducibility. Analysis columns are populated by master_analysis.py
# after generation is complete.
cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        persona_id  INTEGER NOT NULL,
        model_name  TEXT    NOT NULL,
        prompt      TEXT,
        story_text  TEXT,

        -- P1: Atmospheric Sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
        -- Sentence-level RoBERTa sentiment, averaged across the full story.
        sentiment_score      REAL,   -- mean sentiment score (-1 negative → +1 positive)
        sentiment_volatility REAL,   -- std dev of sentence scores (emotional range)

        -- P2: Three-Act Agency Arc (spaCy en_core_web_lg, dependency parsing)
        -- Measures whether the persona drives events or is acted upon, split into
        -- three equal acts. agency_arc = agency_end - agency_start (-1 → +1).
        agency_start  REAL,  -- nsubj ratio in Act I  (opening third)
        agency_middle REAL,  -- nsubj ratio in Act II (middle third)
        agency_end    REAL,  -- nsubj ratio in Act III (final third)
        agency_arc    REAL,  -- trajectory: positive = character gains agency

        -- P3: Agency Quality (spaCy + GloVe en_core_web_lg 300-dim)
        -- For each verb where the persona is subject, measures cosine similarity
        -- to EXPERT vs ROUTINE GloVe centroids. Score = expert/(expert+routine).
        agency_quality_score REAL,  -- range 0 (routine actions) → 1 (expert actions)

        -- P4: Semantic Concreteness (GloVe en_core_web_lg 300-dim)
        -- Scores every content word (noun + verb) in the story against the same
        -- EXPERT/ROUTINE centroids. Story-wide vocabulary register, not just
        -- persona-performed verbs (contrast with P3).
        concreteness_score REAL,    -- range 0 (routine vocabulary) → 1 (expert vocabulary)

        -- P5: Thematic Taxonomy — BART zero-shot (facebook/bart-large-mnli)
        -- Attempted 6-label thematic classification. Collapsed to "Professional
        -- Authority" for 70.1% of all stories — negative result, reported fully.
        taxonomy_label TEXT,        -- dominant BART label assigned to this story

        FOREIGN KEY (persona_id) REFERENCES personas(id)
    )
''')

# ── Populate 75 Personas ───────────────────────────────────────────────────────
# Insertion order is fixed. All generator scripts and merge_to_master.py rely
# on this exact ordering — do not change without regenerating all data.
NATIONALITIES = ['Indian', 'German', 'Nigerian', 'Chinese', 'American']
GENDERS       = ['Male', 'Female', 'Non-Binary']
PROFESSIONS   = ['Engineer', 'Doctor', 'Nurse', 'Teacher', 'Janitor']

cursor.execute("SELECT COUNT(*) FROM personas")
if cursor.fetchone()[0] == 0:
    print("Populating 75 personas (5 nationalities × 3 genders × 5 professions)...")
    for nationality in NATIONALITIES:
        for gender in GENDERS:
            for profession in PROFESSIONS:
                cursor.execute(
                    "INSERT INTO personas (nationality, gender, profession) VALUES (?, ?, ?)",
                    (nationality, gender, profession)
                )
    conn.commit()
    print("75 personas inserted successfully.")
else:
    print(f"Personas already exist — skipping insertion.")

conn.close()
print("Database initialisation complete. Run a generator script next.")