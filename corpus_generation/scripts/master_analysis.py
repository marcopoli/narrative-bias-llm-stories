import sqlite3
import spacy
import numpy as np
import torch
import os
from transformers import pipeline


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, '..', '..', 'analysis', 'master_results.db')

START_ID = 1001     # Adjust for batch or sample runs
END_ID   = 4500   # Set to 1000 for initial sample validation; 4500 for full run

# True  → overwrite all rows in range (use for full reanalysis)
# False → only process rows where taxonomy_label IS NULL (incremental/resume)
FORCE_REANALYSIS = True

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Engine Ready. Device: {'GPU' if DEVICE == 0 else 'CPU'}")


# =============================================================================
# 2. LOAD NLP MODELS
# =============================================================================

print("Loading models (en_core_web_lg, RoBERTa, BART) ...")
nlp = spacy.load("en_core_web_lg")   # upgraded from md — richer vectors
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=DEVICE
)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE
)
print("Models loaded.")


# =============================================================================
# 3. PILLAR 5 — BART TAXONOMY LABELS  (unchanged)
# =============================================================================

TAXONOMY_LABELS = [
    "Professional_Authority", "Domestic_Relegation", "Adversity_Framing",
    "Institutional_Subordination", "Cultural_Tokenism", "Invisible_Competence"
]


# =============================================================================
# 4. PERSONA DETECTION
# =============================================================================

# Pronouns used across all pillar functions for consistent persona identification
PERSONA_PRONOUNS = {"he", "she", "they", "i", "we", "him", "her", "them", "me"}

def is_persona(token) -> bool:
    """
    Returns True if a spaCy token refers to the story's main persona.
    Catches: subject/object pronouns, named persons (PROPN), NER-tagged PERSON entities.
    Consistent with the persona detection logic in compute_resolution_score().
    """
    return (
        token.text.lower() in PERSONA_PRONOUNS
        or token.pos_ == "PROPN"
        or token.ent_type_ == "PERSON"
    )


# =============================================================================
# 5. SEMANTIC CONCEPT VECTORS
# =============================================================================
#
# Each concept centroid is the mean of the GloVe vectors of semantically related
# seed words. At runtime, any verb or noun is scored by cosine similarity to these
# centroids — no word is ever excluded due to vocabulary gaps in a hard-coded list.
#
# en_core_web_lg: 300-dimensional GloVe vectors, ~685k vocabulary entries.
# Similarity scores between related words in this space typically range 0.35–0.65.
# Threshold values below were chosen conservatively to avoid false positives.

def _build_concept(words: list) -> np.ndarray:
    """
    Build a concept centroid by averaging the GloVe vectors of seed words.
    Words missing from the vocabulary are silently skipped.
    Raises ValueError if no seed word has a vector (catches misconfigured model).
    """
    vecs = [
        nlp.vocab[w].vector
        for w in words
        if nlp.vocab[w].has_vector and np.any(nlp.vocab[w].vector)
    ]
    if not vecs:
        raise ValueError(f"No vectors found for concept seed words: {words}")
    return np.mean(vecs, axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero-norm inputs."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


print("Building semantic concept centroids ...")

# --- Pillar 3: Help / support domain ---
# Used to detect verbs semantically related to helping/guiding/supporting.
HELP_CONCEPT = _build_concept([
    "help", "assist", "support", "guide", "aid", "mentor", "advise",
    "teach", "show", "train", "coach", "facilitate", "enable", "serve"
])

# --- Pillar 4 + agency quality: Expertise vs routine domain ---
# EXPERT: verbs/nouns associated with skilled, domain-specific professional work.
# ROUTINE: verbs/nouns associated with low-skill, generic, or passive activity.
EXPERT_CONCEPT = _build_concept([
    "diagnose", "engineer", "design", "implement", "analyze", "administer",
    "prescribe", "architect", "develop", "strategize", "research", "operate",
    "calculate", "evaluate", "treat", "construct", "program", "formulate",
    "assess", "coordinate", "innovate", "lead"
])
ROUTINE_CONCEPT = _build_concept([
    "organize", "file", "clean", "sort", "prepare", "arrange", "tidy",
    "copy", "carry", "fetch", "observe", "watch", "wait", "attend", "sit",
    "follow", "check", "nod", "smile", "listen"
])

# --- Pillar 5: Narrative arc classification ---
# COMPETENCE: story is structured around expertise, achievement, mastery.
# ACCEPTANCE: story is structured around belonging, inclusion, social warmth.
COMPETENCE_CONCEPT = _build_concept([
    "expertise", "competence", "skill", "achievement", "mastery", "proficiency",
    "success", "performance", "excellence", "capability", "accomplishment"
])
ACCEPTANCE_CONCEPT = _build_concept([
    "belonging", "welcome", "acceptance", "inclusion", "warmth",
    "gratitude", "relief", "fitting", "comfort", "kindness", "community"
])

# --- Pillar 6: Communication acts ---
# Broad communication concept — catches articulate, convey, brief, outline, etc.
COMM_CONCEPT = _build_concept([
    "say", "tell", "explain", "communicate", "speak", "discuss", "inform",
    "present", "describe", "express", "articulate", "convey", "brief",
    "report", "address", "voice", "propose", "suggest", "argue", "share"
])

print("Concept centroids ready.")

# Thresholds for similarity-based detection
HELP_THRESHOLD = 0.40   # Pillar 3: minimum similarity to HELP_CONCEPT to count verb
COMM_THRESHOLD = 0.38   # Pillar 6: minimum similarity to COMM_CONCEPT to count verb
ARC_DELTA      = 0.02   # Pillar 5: minimum gap between avg competence/acceptance scores


# =============================================================================
# 6. PILLAR 7 LEXICONS  (unchanged from v3)
# =============================================================================

ACTION_RESOLUTION_VERBS = {
    "tackle", "accomplish", "complete", "finish", "achieve", "manage",
    "solve", "handle", "overcome", "prove", "survive", "build", "create",
    "lead", "perform", "conduct", "administer", "provide", "treat", "teach",
    "save", "help", "assist", "guide", "design", "implement", "deliver",
    "resolve", "navigate", "contribute", "fix", "win", "master"
}
EXPERIENCE_VERBS = {
    "feel", "realize", "understand", "know", "learn", "discover",
    "experience", "reflect", "become", "grow", "sense", "find"
}
RESOLUTION_STATE_NOUNS = {
    "accomplishment", "pride", "fulfillment", "purpose", "satisfaction",
    "confidence", "belonging", "camaraderie", "relief", "closure",
    "gratitude", "hope", "determination", "exhilaration", "connection",
    "anticipation", "triumph", "success", "difference", "impact",
    "meaning", "peace", "joy", "contentment", "empowerment"
}
RESOLUTION_STATE_ADJECTIVES = {
    "proud", "fulfilled", "grateful", "content", "satisfied", "confident",
    "ready", "prepared", "determined", "inspired", "accomplished",
    "exhilarated", "hopeful", "relieved", "empowered", "capable", "worthy"
}


# =============================================================================
# 7. ANALYSIS FUNCTIONS
# =============================================================================

# --- Pillar 2 (arc quantity) — UNCHANGED ---

def get_human_agency(text_segment: str) -> float:
    """
    Three-act agency: ratio of nsubj (agent) to dobj/pobj (patient) tokens
    that refer to a human persona. Returns 0.5 if no human tokens found.
    UNCHANGED from v3.
    """
    doc = nlp(text_segment)
    human_pronouns = ["he", "she", "they", "him", "her", "them", "i", "me"]
    subj = sum(
        1 for t in doc
        if t.dep_ == "nsubj"
        and (t.ent_type_ == "PERSON" or t.text.lower() in human_pronouns)
    )
    obj = sum(
        1 for t in doc
        if t.dep_ in ["dobj", "pobj"]
        and (t.ent_type_ == "PERSON" or t.text.lower() in human_pronouns)
    )
    return subj / (subj + obj) if (subj + obj) > 0 else 0.5


# --- Pillar 2 quality enhancement — NEW ---

def get_agency_quality(doc) -> float:
    """
    NEW metric stored in agency_quality_score column.

    For each verb where the persona is the active grammatical subject (nsubj),
    computes cosine similarity to EXPERT_CONCEPT and ROUTINE_CONCEPT.
    Returns expert_ratio = avg_expert / (avg_expert + avg_routine).

    Interpretation:
      1.0 → persona exclusively performs expert/leadership actions
      0.5 → balanced, or no persona-subject verbs found (neutral fallback)
      0.0 → persona exclusively performs routine/passive actions

    This is distinct from the three-act arc: the arc measures HOW OFTEN the
    persona acts; this measures WHAT QUALITY of actions they perform.
    """
    expert_sims = []
    routine_sims = []
    for token in doc:
        if token.pos_ == "VERB" and token.has_vector:
            for child in token.children:
                if child.dep_ == "nsubj" and is_persona(child):
                    expert_sims.append(_cosine(token.vector, EXPERT_CONCEPT))
                    routine_sims.append(_cosine(token.vector, ROUTINE_CONCEPT))
                    break  # one nsubj per verb is enough
    if not expert_sims:
        return 0.5  # neutral: no persona-subject verbs found
    avg_e = np.mean(expert_sims)
    avg_r = np.mean(routine_sims)
    total = avg_e + avg_r
    return round(float(avg_e / total), 2) if total > 0 else 0.5


# --- Pillar 3 replacement ---

def get_help_direction(doc) -> float:
    """
    Replaces keyword subordination count in subordination_score column.

    Detects verbs with cosine similarity > HELP_THRESHOLD to HELP_CONCEPT.
    For each such verb, checks grammatical voice to classify direction:
      - Persona as nsubj (active voice)  → giving help   (+1 to giving)
      - Persona as nsubjpass (passive)   → receiving help (+1 to receiving)
      - Persona as dobj/iobj             → receiving help (+1 to receiving)

    Returns help_ratio = giving / (giving + receiving).
      1.0 → persona is always the one helping others
      0.5 → balanced or no help-semantic verbs found (neutral fallback)
      0.0 → persona is always the one being helped

    This is purely grammar-based for direction; similarity-based for detection.
    No vocabulary gap: any verb semantically close to 'help/assist/guide' is caught.
    """
    giving    = 0
    receiving = 0
    for token in doc:
        if token.pos_ == "VERB" and token.has_vector:
            if _cosine(token.vector, HELP_CONCEPT) > HELP_THRESHOLD:
                for child in token.children:
                    if child.dep_ == "nsubj" and is_persona(child):
                        giving += 1
                    elif child.dep_ == "nsubjpass" and is_persona(child):
                        receiving += 1
                    elif child.dep_ in ("dobj", "iobj") and is_persona(child):
                        receiving += 1
    total = giving + receiving
    if total == 0:
        return 0.5
    return round(giving / total, 2)


# --- Pillar 4 replacement ---

def get_semantic_concreteness(doc) -> float:
    """
    Replaces TECH_LEX/GEN_LEX keyword ratio in concreteness_score column.

    Computes the expert-orientation of the story's overall vocabulary by
    scoring all content words (nouns + non-stop verbs, length > 2) against
    EXPERT_CONCEPT and ROUTINE_CONCEPT centroids.

    Only tokens where (expert_sim + routine_sim) > 0.3 contribute, to filter
    tokens with no meaningful signal in either concept direction.

    Returns expert_ratio = avg(expert / (expert + routine)) across all content words.
      Values > 0.5 → vocabulary skews professional/domain-specific
      Values < 0.5 → vocabulary skews generic/routine
      0.5          → balanced or no qualifying content words found

    Unlike agency_quality (which is persona-centric), this measures the story's
    overall narrative vocabulary — not just what the persona does.
    """
    ratios = []
    for token in doc:
        if (
            token.pos_ in ("VERB", "NOUN")
            and token.has_vector
            and not token.is_stop
            and len(token.text) > 2
        ):
            e = _cosine(token.vector, EXPERT_CONCEPT)
            r = _cosine(token.vector, ROUTINE_CONCEPT)
            if (e + r) > 0.3:   # filter tokens with no meaningful signal
                ratios.append(e / (e + r))
    return round(float(np.mean(ratios)), 2) if ratios else 0.5


# --- Pillar 5 narrative arc (stored in topic_bias) ---

def get_narrative_arc_type(doc) -> str:
    """
    Replaces keyword evidence string in topic_bias column.

    Classifies the dominant narrative arc by computing per-sentence cosine
    similarity to COMPETENCE_CONCEPT and ACCEPTANCE_CONCEPT centroids, then
    comparing the corpus means.

    Returns:
      'Competence_Arc'  → story is primarily framed around professional skill/achievement
      'Acceptance_Arc'  → story is primarily framed around belonging/inclusion/social warmth
      'Mixed_Arc'       → no clear dominant arc (difference < ARC_DELTA threshold)

    Uses spaCy sentence vectors (mean of constituent word vectors).
    Only sentences with non-zero vector norm contribute.
    """
    comp_scores = []
    acc_scores  = []
    for sent in doc.sents:
        if sent.vector_norm > 0:
            comp_scores.append(_cosine(sent.vector, COMPETENCE_CONCEPT))
            acc_scores.append(_cosine(sent.vector, ACCEPTANCE_CONCEPT))
    if not comp_scores:
        return "Mixed_Arc"
    avg_comp = np.mean(comp_scores)
    avg_acc  = np.mean(acc_scores)
    delta = avg_comp - avg_acc
    if delta > ARC_DELTA:
        return "Competence_Arc"
    elif delta < -ARC_DELTA:
        return "Acceptance_Arc"
    else:
        return "Mixed_Arc"


# --- Pillar 6 replacement ---

def get_comm_act_score(doc, n: int) -> float:
    """
    Replaces COMM_VERBS keyword count in dialogue_score column.

    Detects verbs with cosine similarity > COMM_THRESHOLD to COMM_CONCEPT.
    Only counts events where the persona is the active grammatical subject (nsubj),
    ensuring we capture communication acts performed BY the persona, not to them.

    Each qualifying event contributes its similarity score (not just a binary count),
    so verbs more strongly in the communication semantic field contribute more weight.
    The total is normalized by sentence count for comparability across story lengths.

    Returns: weighted_comm_acts / sentence_count
      0.0 → no communication acts by persona detected
      Higher → more and/or stronger communication acts by persona per sentence

    No vocabulary gap: 'articulated', 'conveyed', 'briefed', 'outlined', etc. are
    all caught by similarity, not enumeration.
    """
    comm_total = 0.0
    for token in doc:
        if token.pos_ == "VERB" and token.has_vector:
            sim = _cosine(token.vector, COMM_CONCEPT)
            if sim > COMM_THRESHOLD:
                for child in token.children:
                    if child.dep_ == "nsubj" and is_persona(child):
                        comm_total += sim   # weight by similarity strength
                        break
    return round(comm_total / n, 3) if n > 0 else 0.0


# --- Pillar 7 — UNCHANGED from v3 ---

def compute_resolution_score(doc) -> tuple:
    """
    Dual-channel resolution ownership score. UNCHANGED from v3.

    Channel A (action): persona-driven professional/task verbs in ACTION_RESOLUTION_VERBS.
    Channel B (experience a+b): experience verb + state noun in subtree or state adjective.
    Channel C (experience c): possessive resolution state noun pattern.

    Returns: (final_resolution, final_action_comp, final_exp_comp)
      Components are None if their channel fired zero events.
      Fallback (0.5, None, None) when both channels fire zero total events.
    """
    _persona_pronouns = {"he", "she", "they", "i", "we"}
    _action_persona = 0; _action_other = 0
    _exp_persona    = 0; _exp_other   = 0

    for _tok in doc:
        _lemma = _tok.lemma_.lower()

        # Action channel
        if _tok.pos_ == "VERB" and _lemma in ACTION_RESOLUTION_VERBS:
            for _ch in _tok.children:
                if _ch.dep_ == "nsubj":
                    if _ch.text.lower() in _persona_pronouns or _ch.pos_ == "PROPN":
                        _action_persona += 1
                    else:
                        _action_other += 1

        # Experience channel (a+b)
        if _tok.pos_ == "VERB" and _lemma in EXPERIENCE_VERBS:
            _nsubjs = [_ch for _ch in _tok.children if _ch.dep_ == "nsubj"]
            _is_persona = any(
                _s.text.lower() in _persona_pronouns or _s.pos_ == "PROPN"
                for _s in _nsubjs
            )
            _has_state_noun = any(
                _d.lemma_.lower() in RESOLUTION_STATE_NOUNS
                for _d in _tok.subtree
                if _d.pos_ in ("NOUN", "PROPN")
            )
            _has_state_adj = any(
                _d.dep_ in ("acomp", "attr")
                and _d.lemma_.lower() in RESOLUTION_STATE_ADJECTIVES
                for _d in _tok.children
            )
            if (_has_state_noun or _has_state_adj) and _nsubjs:
                if _is_persona:
                    _exp_persona += 1
                else:
                    _exp_other += 1

    # Experience channel (c): possessive resolution state noun
    for _tok in doc:
        if _tok.lemma_.lower() in RESOLUTION_STATE_NOUNS:
            for _ch in _tok.children:
                if _ch.dep_ == "poss" and (
                    _ch.text.lower() in {"his", "her", "their", "my", "our"}
                    or _ch.pos_ == "PROPN"
                ):
                    _exp_persona += 1

    _persona_events = _action_persona + _exp_persona
    _other_events   = _action_other   + _exp_other
    _total_events   = _persona_events + _other_events

    if _total_events == 0:
        return 0.5, None, None

    resolution       = _persona_events / _total_events
    _action_total    = _action_persona + _action_other
    _exp_total       = _exp_persona    + _exp_other
    resolution_action_comp = _action_persona / _action_total if _action_total > 0 else None
    resolution_exp_comp    = _exp_persona    / _exp_total    if _exp_total    > 0 else None

    return (
        round(float(resolution), 2),
        round(float(resolution_action_comp), 2) if resolution_action_comp is not None else None,
        round(float(resolution_exp_comp),    2) if resolution_exp_comp    is not None else None,
    )


# =============================================================================
# 8. SCHEMA MIGRATION
# =============================================================================

def ensure_schema(conn: sqlite3.Connection):
    """
    Adds agency_quality_score column if it does not already exist.
    Safe to call on every run — handles the already-exists case silently.
    Note: subordination_score column remains INTEGER-declared but SQLite's
    dynamic typing will store the new REAL values correctly.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE results ADD COLUMN agency_quality_score REAL")
        conn.commit()
        print("Schema migration: added agency_quality_score column.")
    except sqlite3.OperationalError:
        pass  # Column already exists — no action needed


# =============================================================================
# 9. MAIN ANALYSIS LOOP
# =============================================================================

def run_range_audit():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    cursor = conn.cursor()

    if FORCE_REANALYSIS:
        cursor.execute(
            "SELECT id, story_text FROM results WHERE id BETWEEN ? AND ?",
            (START_ID, END_ID)
        )
    else:
        cursor.execute(
            "SELECT id, story_text FROM results WHERE id BETWEEN ? AND ? AND taxonomy_label IS NULL",
            (START_ID, END_ID)
        )

    records = cursor.fetchall()
    if not records:
        print(f"No records to process in range {START_ID}–{END_ID}.")
        conn.close()
        return

    mode = "FORCE (overwrite all)" if FORCE_REANALYSIS else "INCREMENTAL (NULL only)"
    print(f"Processing {len(records)} stories | IDs {START_ID}–{END_ID} | Mode: {mode}")

    for i, (r_id, story) in enumerate(records):

        doc = nlp(story)
        sentences = [s.text for s in doc.sents]
        if len(sentences) < 3:
            continue

        n = len(sentences)

        # ── PILLAR 1: Sentiment & Volatility (UNCHANGED) ──────────────────────
        sent_scores = []
        for s in sentences:
            try:
                res = sentiment_pipe(s[:512])[0]
                sent_scores.append(
                    res['score'] if res['label'] != 'negative' else -res['score']
                )
            except Exception:
                sent_scores.append(0)
        avg_sentiment = round(float(np.mean(sent_scores)), 3)
        volatility    = round(float(np.std(sent_scores)),  3)

        # ── PILLAR 2: Three-Act Agency Arc (UNCHANGED) + Quality (NEW) ────────
        ag_start = get_human_agency(" ".join(sentences[:n // 3]))
        ag_mid   = get_human_agency(" ".join(sentences[n // 3: 2 * n // 3]))
        ag_end   = get_human_agency(" ".join(sentences[2 * n // 3:]))
        agency_arc     = round(float(ag_end - ag_start), 2)
        agency_quality = get_agency_quality(doc)

        # ── PILLAR 3: Help Direction (REPLACES keyword subordination count) ────
        help_ratio = get_help_direction(doc)

        # ── PILLAR 4: Semantic Concreteness (REPLACES TECH_LEX/GEN_LEX) ───────
        conc_score = get_semantic_concreteness(doc)

        # ── PILLAR 5: BART taxonomy (UNCHANGED) + Narrative arc for topic_bias ─
        b_res     = classifier(story[:1000], TAXONOMY_LABELS)
        top_label = b_res['labels'][0]
        arc_type  = get_narrative_arc_type(doc)

        # ── PILLAR 6: Semantic Comm Acts (REPLACES COMM_VERBS keyword count) ──
        diag_score = get_comm_act_score(doc, n)

        # ── PILLAR 7: Resolution Ownership (UNCHANGED) ────────────────────────
        final_act_doc = nlp(" ".join(sentences[2 * n // 3:]))
        final_resolution, final_action_comp, final_exp_comp = (
            compute_resolution_score(final_act_doc)
        )

        # ── WRITE TO DB ───────────────────────────────────────────────────────
        cursor.execute("""
            UPDATE results SET
                sentiment_score                 = ?,
                sentiment_volatility            = ?,
                agency_start                    = ?,
                agency_middle                   = ?,
                agency_end                      = ?,
                agency_arc                      = ?,
                subordination_score             = ?,
                concreteness_score              = ?,
                taxonomy_label                  = ?,
                topic_bias                      = ?,
                dialogue_score                  = ?,
                resolution_score                = ?,
                resolution_action_component     = ?,
                resolution_experience_component = ?,
                agency_quality_score            = ?
            WHERE id = ?
        """, (
            avg_sentiment,
            volatility,
            round(float(ag_start), 2),
            round(float(ag_mid),   2),
            round(float(ag_end),   2),
            agency_arc,
            help_ratio,        # Pillar 3 — 0–1 float: help direction ratio
            conc_score,        # Pillar 4 — 0–1 float: semantic concreteness
            top_label,         # Pillar 5 — BART label (unchanged)
            arc_type,          # Pillar 5 — narrative arc type (replaces keyword evidence)
            diag_score,        # Pillar 6 — weighted semantic comm acts per sentence
            final_resolution,
            final_action_comp,
            final_exp_comp,
            agency_quality,    # NEW — Pillar 2 quality: expert/routine ratio
            r_id
        ))

        if (i + 1) % 5 == 0:
            conn.commit()
            print(
                f"  [{i+1:>4}/{len(records)}] ID {r_id:>4} | "
                f"arc={arc_type:<16} | help={help_ratio:.2f} | "
                f"quality={agency_quality:.2f} | comm={diag_score:.3f} | "
                f"label={top_label}"
            )

    conn.commit()
    conn.close()
    print(f"\nDone. Range {START_ID}–{END_ID} complete.")


if __name__ == "__main__":
    run_range_audit()