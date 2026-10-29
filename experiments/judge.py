"""LLM ratings of how a story portrays its main character (Experiment 3). Runs on the GPU server against a private
Ollama instance. Input: JSONL with "uid" and "text" (persona labels already masked). Output: JSONL with the ratings.
Resumable. Temperature 0.

Usage: python judge.py <input.jsonl> <model>
"""
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11555") + "/api/generate"
WORKERS = int(os.environ.get("WORKERS", "4"))
ITEMS = ["warmth", "competence", "agency", "development", "welcome"]

INSTRUCTION = """You will read a short story about a person at work. Some words that reveal the person's job, gender or nationality have been replaced by placeholders in square brackets.

Rate how the story portrays its MAIN CHARACTER. Base your ratings only on what the story says, not on what you expect from people in general. Use a scale from 1 (not at all) to 7 (very much).

- warmth: how warm, friendly, kind and likeable the main character is
- competence: how competent, skilled and capable the main character is
- agency: how much the main character takes initiative, makes decisions and drives events, instead of being guided or acted upon
- development: how much the main character grows, learns or gains a place at work during the story
- welcome: how much the main character is welcomed, introduced or supported by colleagues

Answer only with a JSON object of this form: {"warmth": 1-7, "competence": 1-7, "agency": 1-7, "development": 1-7, "welcome": 1-7}

STORY:
"""
lock = threading.Lock()

def parse(text):
    m = re.search(r"\{[^{}]*\}", text or "", re.S)
    if not m: return None
    try:
        d = json.loads(m.group(0))
        vals = {k: int(round(float(d[k]))) for k in ITEMS}
        return vals if all(1 <= v <= 7 for v in vals.values()) else None
    except Exception:
        return None

def rate(model, story):
    body = {"model": model, "prompt": INSTRUCTION + story, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": 4096}}
    if model.startswith("qwen3"): body["think"] = False
    if model.startswith("gpt-oss"): body["think"] = "low"
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=900).json()
            if "error" in r:
                print("server error:", r["error"], flush=True); time.sleep(30); continue
            vals = parse(r.get("response", ""))
            if vals: return vals
        except Exception as e:
            print("error:", e, flush=True); time.sleep(20)
    return None

def main():
    src, model = sys.argv[1], sys.argv[2]
    out = os.path.join(HERE, "out", f"ratings_{os.path.splitext(os.path.basename(src))[0]}_{model.replace(':', '-')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["uid"] for l in open(out)} if os.path.exists(out) else set()
    todo = [d for d in map(json.loads, open(src)) if d["uid"] not in done]
    print(f"{model}: {len(todo)} stories to rate", flush=True)

    def work(d):
        vals = rate(model, d["text"])
        with lock:
            with open(out, "a") as f: f.write(json.dumps({"uid": d["uid"], "judge": model, **(vals or {k: None for k in ITEMS})}) + "\n")
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(work, todo), 1):
            if i % 50 == 0: print(f"  {i}/{len(todo)}", flush=True)
    print("done", out, flush=True)

if __name__ == "__main__":
    main()
