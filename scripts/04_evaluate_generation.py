"""Generation-based evaluation of a trained run. All metrics are computed after decoding to
Koma53, so they are comparable across pitch encodings, and each is also computed on the real
test continuations as a reference value.

  python scripts/04_evaluate_generation.py runs/transformer-koma53_rel-prefix-s1
  python scripts/04_evaluate_generation.py reference      # reference values only, no model

Metrics (prompt = conditioning prefix + first PROMPT notes of a test piece; the model continues):
  scale_overlap   histogram intersection between the generated pitch-class distribution
                  (53 commas, relative to karar) and the makam's training distribution
  off_scale_rate  share of generated notes on pitch classes that are rare (<0.5%) in the makam
  bar_validity    share of complete generated measures whose durations sum to the usul's measure length
  copy_rate       share of generated 8-note (pitch, duration) n-grams that occur in the training set
  grammar_ok      share of generated token groups that follow the [BAR] P D pattern
"""
import argparse, collections, json, os, sys
from fractions import Fraction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src import dataset as D
from src.models import build_model, generate
from src.symbtr import REST
from src.tokenizer import Vocab, decode_pitch, is_bar_start, BAR

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("--prompt", type=int, default=16)
p.add_argument("--new_notes", type=int, default=64); p.add_argument("--max_pieces", type=int, default=0)
p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
REF_ONLY = a.run == "reference"       # reference values only, no model needed
if REF_ONLY:
    cfg = dict(pitch="koma53", cond="prefix")
else:
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    vj = json.load(open(os.path.join(a.run, "vocab.json")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

corpus = D.load_corpus()
karar = D.karar_by_makam(corpus["train"]); ae2k = D.ae_to_koma(corpus["train"])
if cfg.get("smoke"): corpus = {k: v[::max(len(v) // 40, 1)] for k, v in corpus.items()}
D.tokenize_corpus(corpus, cfg["pitch"], karar, cfg["cond"] == "prefix")
vocab = D.build_vocab(corpus["train"]) if REF_ONLY else Vocab(vj["itos"])
D.encode_corpus(corpus, vocab)
if not REF_ONLY:
    model = build_model(cfg["model"], len(vocab), cfg).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=dev)); model.eval()

# ---------- reference statistics from the training set
N = 8
pc_ref = collections.defaultdict(collections.Counter)      # makam -> pitch-class histogram
ngrams = set(); mlen = collections.defaultdict(collections.Counter)
for x in corpus["train"]:
    k = karar[x["meta"]["makam"]]; ev = x["events"]
    seq = [(e["koma"], Fraction(e["num"], e["den"])) for e in ev]
    for e in ev:
        if e["koma"] != REST: pc_ref[x["meta"]["makam"]][(e["koma"] - k) % 53] += 1
    for i in range(len(seq) - N + 1): ngrams.add(tuple(seq[i:i + N]))
    bars = [i for i, e in enumerate(ev) if is_bar_start(e["onset"])]
    for s, e in zip(bars, bars[1:]):                   # measure length = notated duration between bar starts
        mlen[x["meta"]["usul"]][sum(q[1] for q in seq[s:e])] += 1
measure_len = {u: c.most_common(1)[0][0] for u, c in mlen.items()}
def norm(c):
    s = sum(c.values()) or 1; return {k: v / s for k, v in c.items()}
pc_ref = {m: norm(c) for m, c in pc_ref.items()}


def parse_groups(tokens, makam):
    """Token list -> list of (bar_flag, koma, Fraction) plus the number of malformed groups."""
    notes, bad, i, k = [], 0, 0, karar.get(makam)
    while i < len(tokens):
        bar = tokens[i] == BAR
        j = i + 1 if bar else i
        if j + 1 < len(tokens) and tokens[j].startswith("P:") and tokens[j + 1].startswith("D:"):
            koma = decode_pitch(tokens[j], cfg["pitch"], k, ae2k)
            num, den = tokens[j + 1][2:].split("/")
            if koma is None: bad += 1
            else: notes.append((bar, koma, Fraction(int(num), int(den))))
            i = j + 2
        else:
            bad += 1; i += 1
    return notes, bad


def stats():
    return dict(pc=collections.defaultdict(collections.Counter), bars=[0, 0], ng=[0, 0], grp=[0, 0], off=[0, 0])

def update(st, notes, bad, makam, usul):
    ref = pc_ref.get(makam, {}); k = karar.get(makam)
    for _, koma, _ in notes:
        if koma == REST: continue
        pc = (koma - k) % 53; st["pc"][makam][pc] += 1
        st["off"][1] += 1; st["off"][0] += ref.get(pc, 0.0) < 0.005
    st["grp"][0] += len(notes); st["grp"][1] += len(notes) + bad
    # complete measures: between two BAR flags
    idx = [i for i, n in enumerate(notes) if n[0]]
    for s, e in zip(idx, idx[1:]):
        st["bars"][1] += 1
        st["bars"][0] += sum(n[2] for n in notes[s:e]) == measure_len.get(usul)
    seq = [(n[1], n[2]) for n in notes]
    for i in range(len(seq) - N + 1):
        st["ng"][1] += 1; st["ng"][0] += tuple(seq[i:i + N]) in ngrams

def finish(st):
    ov, w = 0.0, 0
    for m, c in st["pc"].items():
        g = norm(c); n = sum(c.values())
        ov += n * sum(min(g.get(i, 0), pc_ref.get(m, {}).get(i, 0)) for i in range(53)); w += n
    r = lambda x: x[0] / x[1] if x[1] else None
    return dict(scale_overlap=ov / w if w else None, off_scale_rate=r(st["off"]), bar_validity=r(st["bars"]),
                copy_rate=r(st["ng"]), grammar_ok=r(st["grp"]), notes=st["off"][1], measures=st["bars"][1])

gen, real = stats(), stats()
pieces = [x for x in corpus["test"] if len(x["groups"]) >= a.prompt + a.new_notes]
if a.max_pieces: pieces = pieces[:a.max_pieces]
for n, x in enumerate(pieces):
    m, u = x["meta"]["makam"], x["meta"]["usul"]
    ref_tokens = sum(x["groups"][a.prompt:a.prompt + a.new_notes], [])
    update(real, *parse_groups(ref_tokens, m), m, u)
    if REF_ONLY: continue
    cut = x["note_starts"][a.prompt]
    head = [vocab.bos] + x["prefix_ids"] + x["ids"][:cut]
    is_pitch = lambda i: vocab.itos[i].startswith("P:")
    stop = lambda new: sum(map(is_pitch, new)) >= a.new_notes and vocab.itos[new[-1]].startswith("D:")
    new = generate(model, head, a.new_notes * 4, vocab.eos, a.temperature, a.top_p, stop)
    update(gen, *parse_groups(vocab.decode(new), m), m, u)
    if (n + 1) % 50 == 0: print(f"{n + 1}/{len(pieces)}", flush=True)
res = dict(run=os.path.basename(a.run.rstrip("/")), pieces=len(pieces), prompt_notes=a.prompt, new_notes=a.new_notes,
           temperature=a.temperature, top_p=a.top_p, generated=None if REF_ONLY else finish(gen), reference=finish(real))
if not REF_ONLY:
    json.dump(res, open(os.path.join(a.run, "generation_metrics.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
