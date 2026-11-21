"""Collects every result file into flat tables under reports/results/ and records where
each file came from. Run from the repo root (needs git history): python scripts/06_collect_results.py

  likelihood.csv   one row per trained run (current and archived rounds)
  generation.csv   one row per generation evaluation file of the current round
  reference.csv    the same metrics computed on real test music
  replicates.csv   repeated trainings of the two countdown models (same seed, different sessions)
"""
import csv, glob, json, os, re, subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "reports", "results"); os.makedirs(OUT, exist_ok=True)

# Result files were pushed after each GPU session; the commit identifies the session and,
# because new pods retrain, the model copy ("checkpoint") that produced the file.
SESSIONS = {
    "42dbf58": ("round3", "A"),      # full grid on corrected data
    "39426c1": ("karar", "B"),       # duration-first test; countdown models retrained
    "fe7220f": ("karar2", "B"),      # loop penalty, corrected ending rule; same pod as karar
    "db10801": ("karar3", "C"),      # ending selection; new pod, retrained
    "7430494": ("karar4", "D"),      # blending; new pod, retrained
}

def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout.strip()

def provenance(path):
    h = git("log", "-1", "--format=%h", "--", os.path.relpath(path, ROOT))
    s = SESSIONS.get(h, ("", ""))
    return h, s[0], s[1]

def cfg_cols(c):
    return dict(model=c["model"], pitch=c["pitch"], cond=c["cond"], pos=int(bool(c.get("pos"))),
                meter=int(bool(c.get("meter"))), countdown=int(bool(c.get("rem"))), order=c.get("order", "pd"),
                ctx=c["ctx"], dropout=c["dropout"], layers=c["layers"] if c["model"] == "transformer" else c["lstm_layers"],
                width=c["d_model"] if c["model"] == "transformer" else c["lstm_hidden"])

def write(name, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(name, len(rows), "rows")

# ---------- likelihood
rows = []
for rnd, pat in (("round3", "runs/*-s1"), ("round2_archived", "runs/archive_v2/*"), ("round1_archived", "runs/archive_v1/*")):
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        f = os.path.join(d, "metrics.json")
        if not os.path.exists(f): continue
        m = json.load(open(f)); h, sess, ck = provenance(f)
        best = min(m["history"], key=lambda x: x["nll_per_token"])
        r = dict(round=rnd, run=m["name"], **cfg_cols(m["config"]), n_params=m["n_params"], vocab=m["vocab_size"],
                 best_step=best["step"], steps_run=m["steps_run"], train_seconds=m["train_seconds"])
        for split in ("val", "test", "few_shot"):
            if split in m:
                for k in ("bits_per_note", "pitch_bits_per_note", "dur_bits_per_note", "nll_per_token"):
                    r[f"{split}_{k}"] = round(m[split][k], 4)
        r.update(commit=h, session=sess, checkpoint=ck if rnd == "round3" else "")
        rows.append(r)
write("likelihood.csv", rows)

# ---------- generation
gen, refs = [], {}
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "*-s1", "gen_*.json"))):
    j = json.load(open(f)); d = os.path.dirname(f)
    c = json.load(open(os.path.join(d, "config.json"))); h, sess, ck = provenance(f)
    mix = j.get("mix") or {}
    r = dict(run=os.path.basename(d), file=os.path.basename(f), session=sess, checkpoint=ck, commit=h, **cfg_cols(c),
             scenario=2 if j.get("full") else 1, prompt_notes=j["prompt_notes"], constrained=int(bool(j.get("constrained"))),
             copy_penalty=j.get("copy_penalty", 0.0), select=j.get("select", 0), select_measures=j.get("select_measures") or "",
             mixed_with=mix.get("run", ""), mix_body=mix.get("body", ""), mix_end=mix.get("end", ""), pieces=j["pieces"])
    for k, v in (j.get("selection") or {}).items(): r["selection_" + k] = round(v, 4) if isinstance(v, float) else v
    for k, v in j["generated"].items():
        r[k] = round(v, 4) if isinstance(v, float) else json.dumps(v) if isinstance(v, list) else v
    gen.append(r)
    key = (r["scenario"], j["prompt_notes"], "12-TET" if c["pitch"] == "tet12" else "AEU" if c["pitch"] == "aeu" else "53-comma")
    if key not in refs or len(j["reference"]) > len(refs[key]): refs[key] = j["reference"]
write("generation.csv", gen)
write("reference.csv", [dict(scenario=k[0], prompt_notes=k[1], representation=k[2],
                             **{a: round(b, 4) if isinstance(b, float) else json.dumps(b) if isinstance(b, list) else b for a, b in v.items()})
                        for k, v in sorted(refs.items())])

# ---------- repeated trainings (from git history of metrics.json)
rep = []
for run in ("transformer-koma53-prefix-pos-meter-rem-ctx768-s1", "lstm-koma53-prefix-pos-meter-rem-do0.5-s1"):
    path = f"runs/{run}/metrics.json"
    for h in git("log", "--format=%h", "--", path).split():
        try: m = json.loads(git("show", f"{h}:{path}"))
        except json.JSONDecodeError: continue
        s = SESSIONS.get(h, ("", ""))
        rep.append(dict(run=run, commit=h, session=s[0], checkpoint=s[1], device=m["device"], steps_run=m["steps_run"],
                        val_bits_per_note=round(m["val"]["bits_per_note"], 4), test_bits_per_note=round(m["test"]["bits_per_note"], 4),
                        test_pitch_bits=round(m["test"]["pitch_bits_per_note"], 4), test_dur_bits=round(m["test"]["dur_bits_per_note"], 4)))
write("replicates.csv", rep)
