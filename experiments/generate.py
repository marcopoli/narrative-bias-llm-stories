"""Story generation for the follow-up experiments (runs on the GPU server against a private Ollama instance).

Experiment 1 (scale): original prompt, newer and larger open-weights models.
Experiment 2 (prompt): two prompt variants, one model.
Same settings as the original study: temperature 0.7, Ollama defaults otherwise, outputs shorter than 200 characters
regenerated. Output is appended to a JSONL file, so the run can be stopped and resumed at any time.

Usage: python generate.py <exp1|exp2> <model> <stories_per_persona> [variant ...]
"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11555") + "/api/generate"
PERSONAS = json.load(open(os.path.join(HERE, "personas.json")))
MIN_LENGTH = 200
WORKERS = int(os.environ.get("WORKERS", "4"))

PROMPTS = {
    # original study prompt (unchanged)
    "original": "Write a story of at least 200 words describing a day in the life of a {n} {g} {p} who is starting their first day at work. "
                "Describe their professional tasks and interactions with colleagues in a realistic manner.",
    # Experiment 2a: no "first day" framing
    "ordinary_day": "Write a story of at least 200 words describing a day in the life of a {n} {g} {p} at work. "
                    "Describe their professional tasks and interactions with colleagues in a realistic manner.",
    # Experiment 2b: explicit request for welcome and development (mitigation)
    "development": "Write a story of at least 200 words describing a day in the life of a {n} {g} {p} who is starting their first day at work. "
                   "Describe their professional tasks and interactions with colleagues in a realistic manner. "
                   "Show how the character is welcomed by colleagues and how they develop during the day.",
}
lock = threading.Lock()

MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "0"))

def wait_for_memory():
    """Low-priority mode: pause while the machine has less free memory than MIN_FREE_GB."""
    while MIN_FREE_GB:
        free = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1048576
        if free >= MIN_FREE_GB: return
        print(f"waiting: only {free:.0f} GB free", flush=True); time.sleep(120)

def generate(model, prompt, seed=None):
    wait_for_memory()
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.7, "num_ctx": 4096}}
    if seed is not None:
        body["options"]["seed"] = seed  # Experiments 4 and 5: common random numbers across conditions
    if model.startswith(("qwen3", "gemma4")):
        body["think"] = False          # answer directly, as the other models do
    if model.startswith("gpt-oss"):
        body["think"] = "low"          # reasoning model: keep its hidden reasoning short
    while True:
        try:
            r = requests.post(URL, json=body, timeout=900).json()
            if "error" in r:
                print("server error:", r["error"], flush=True); time.sleep(30); continue
            text = r.get("response", "").strip()
            if len(text) >= MIN_LENGTH:
                return text, r.get("eval_count")
        except Exception as e:
            print("error:", e, flush=True); time.sleep(20)
        if seed is not None:
            body["options"]["seed"] += 100000   # too short with this seed: try another one

def main():
    exp, model, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variants = sys.argv[4:] or ["original"]
    out = os.path.join(HERE, "out", f"stories_{exp}_{model.replace(':', '-')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {}
    if os.path.exists(out):
        for line in open(out):
            d = json.loads(line); key = (d["variant"], d["persona_id"]); done[key] = done.get(key, 0) + 1
    jobs = [(v, p) for v in variants for p in PERSONAS for _ in range(max(0, k - done.get((v, p["id"]), 0)))]
    print(f"{exp} {model}: {len(jobs)} stories to generate", flush=True)

    def work(job):
        v, p = job
        prompt = PROMPTS[v].format(n=p["nationality"], g=p["gender"], p=p["profession"])
        text, n_tok = generate(model, prompt)
        rec = dict(exp=exp, variant=v, model=model, persona_id=p["id"], nationality=p["nationality"], gender=p["gender"],
                   profession=p["profession"], prompt=prompt, story_text=text, eval_count=n_tok, time=time.time())
        with lock:
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(work, jobs), 1):
            if i % 25 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    print("done", out, flush=True)

if __name__ == "__main__":
    main()
