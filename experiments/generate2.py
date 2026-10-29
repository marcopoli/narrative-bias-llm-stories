"""Story generation for Experiments 4 and 5 (runs on the GPU server against a private Ollama instance).

Experiment 4 (occupations): the original prompt with 13 occupations, 11 of which have human warmth and competence
norms (Friehs et al., 2022). Every gender x nationality combination is written with the same seeds for all
occupations, so that stories with the same seed differ only in the occupation word (common random numbers).

Experiment 5 (mitigation): four conditions on the 75 original personas, with the same seeds in every condition.
  baseline - the original prompt
  parity   - the original prompt plus an instruction to give the character full agency (instruction-based)
  blind    - a story about an unnamed new employee, then adapted to the persona (decouple plot from label)
  revise   - the baseline story, checked and revised by the same model for the character's agency (self-revision)

Settings as in the original study (temperature 0.7, outputs shorter than 200 characters regenerated). Resumable.

Usage: python generate2.py <exp4|exp5> <model> <stories_per_unit> [variant ...]   (Experiment 5: default all four)
"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from generate import PROMPTS, generate, lock, HERE, PERSONAS, WORKERS

ORIGINAL = PROMPTS["original"]
OCCUPATIONS = ["Engineer", "Doctor", "Nurse", "Teacher", "Janitor", "Banker", "Childcare Worker", "Judge", "Politician",
               "Farmer", "Craftsperson", "Firefighter", "Police Officer"]
NATIONALITIES = ["American", "German", "Indian", "Nigerian", "Chinese"]
GENDERS = ["Male", "Female", "Non-Binary"]

PARITY = (" Give the main character the same initiative, decision-making and competence that you would give any "
          "skilled professional, and avoid stereotypes about their job, gender or nationality.")
BLIND_DRAFT = ("Write a story of at least 200 words describing a day in the life of a person who is starting their first "
               "day at a new job. Describe their tasks and interactions with colleagues in a realistic manner. "
               "Do not mention the kind of job, the person's gender or their nationality.")
BLIND_ADAPT = ("Rewrite the following story so that the main character is a {n} {g} {p}. Keep the events, the decisions "
               "the character makes and the way colleagues treat them. Change only what is needed to make the job "
               "realistic. Answer only with the story.\n\nSTORY:\n{s}")
REVISE = ("Read the following story. Check whether the main character takes initiative, makes decisions and drives the "
          "events as much as a skilled professional would. Revise the story so that they do, while keeping their job "
          "realistic and the story at least 200 words long. Answer only with the revised story.\n\nSTORY:\n{s}")

def gen(model, prompt, seed):
    return generate(model, prompt, seed=seed)

def run(exp, model, k, variants=None):
    out = os.path.join(HERE, "out", f"stories_{exp}_{model.replace(':', '-')}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(d["variant"], d["unit"], d["rep"]) for d in map(json.loads, open(out))}
    if exp == "exp4":
        units = [dict(unit=f"{p}|{g}|{n}", nationality=n, gender=g, profession=p, combo=i)
                 for i, (g, n) in enumerate((g, n) for g in GENDERS for n in NATIONALITIES) for p in OCCUPATIONS]
        jobs = [("original", u, r) for u in units for r in range(k)]
    else:
        units = [dict(unit=str(p["id"]), persona_id=p["id"], nationality=p["nationality"], gender=p["gender"],
                      profession=p["profession"], combo=GENDERS.index(p["gender"]) * 5 + NATIONALITIES.index(p["nationality"]))
                 for p in PERSONAS]
        # the baseline comes first because the revised stories start from it
        jobs = [(v, u, r) for v in (variants or ["baseline", "parity", "blind", "revise"]) for u in units for r in range(k)]
    jobs = [j for j in jobs if (j[0], j[1]["unit"], j[2]) not in done]
    print(f"{exp} {model}: {len(jobs)} stories to generate", flush=True)
    drafts, base = {}, {}
    if os.path.exists(out):
        for d in map(json.loads, open(out)):
            if d["variant"] == "baseline": base[(d["unit"], d["rep"])] = d["story_text"]
            if d.get("draft"): drafts[d["seed"]] = d["draft"]

    def work(job):
        v, u, r = job
        seed = 1000 * (r + 1) + u["combo"]            # same seed for all occupations of a gender x nationality combination
        n, g, p = u["nationality"], u["gender"], u["profession"]
        draft = None
        if v in ("original", "baseline"):
            prompt = ORIGINAL.format(n=n, g=g, p=p)
        elif v == "parity":
            prompt = ORIGINAL.format(n=n, g=g, p=p) + PARITY
        elif v == "blind":
            with lock:
                draft = drafts.get(seed)
            if draft is None:
                draft, _ = gen(model, BLIND_DRAFT, seed)
                with lock:
                    draft = drafts.setdefault(seed, draft)
            prompt = BLIND_ADAPT.format(n=n, g=g, p=p, s=draft)
        else:
            while (u["unit"], r) not in base: time.sleep(30)
            prompt = REVISE.format(s=base[(u["unit"], r)])
        text, n_tok = gen(model, prompt, seed)
        rec = dict(exp=exp, variant=v, model=model, unit=u["unit"], rep=r, seed=seed, persona_id=u.get("persona_id"),
                   nationality=n, gender=g, profession=p, prompt=prompt, story_text=text, eval_count=n_tok,
                   draft=draft, time=time.time())
        with lock:
            if v == "baseline": base[(u["unit"], r)] = text
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(work, jobs), 1):
            if i % 25 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    print("done", out, flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:] or None)
