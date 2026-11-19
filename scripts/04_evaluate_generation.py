"""Generation-based evaluation of a trained run, for the two usage scenarios of the editor.

  Scenario 1, autocomplete: the model continues a piece from its first PROMPT notes.
      python scripts/04_evaluate_generation.py runs/<name> --prompt 16
  Scenario 2, composition from scratch: only makam/usul/form are given; the model writes a
  whole piece until it emits EOS.
      python scripts/04_evaluate_generation.py runs/<name> --prompt 0 --full
  Add --constrain for usul-constrained decoding (see src/generation.py).
  `reference` instead of a run directory prints the reference values only (no model).

All metrics are computed after decoding to Koma53, so they are comparable across pitch
encodings, and each is also computed on the real test music as a reference value.
  scale_overlap   histogram intersection between the generated pitch-class distribution
                  (53 commas, relative to karar) and the makam's training distribution
  off_scale_rate  share of notes on pitch classes that are rare (<0.5%) in the makam
  bar_validity    share of complete measures whose durations sum to the usul's measure length
  copy_rate       share of 8-note (pitch, duration) n-grams that occur in the training set
  grammar_ok      share of token groups that follow the [BAR] [POS] P D pattern
  --full only:
  ends_on_karar   share of pieces whose last sounded note is the karar of the makam
  eos_rate        share of pieces the model ended itself (before the token limit)
  mean_notes      mean piece length in notes
"""
import argparse, collections, json, os, sys
from fractions import Fraction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src import dataset as D
from src.models import build_model
from src.generation import generate_batch
from src.symbtr import REST
from src.tokenizer import Vocab, decode_pitch, BAR

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("--prompt", type=int, default=16)
p.add_argument("--new_notes", type=int, default=64); p.add_argument("--max_pieces", type=int, default=0)
p.add_argument("--full", action="store_true", help="generate until EOS instead of --new_notes")
p.add_argument("--max_tokens", type=int, default=3000, help="token limit for --full")
p.add_argument("--constrain", action="store_true", help="usul-constrained decoding")
p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--batch", type=int, default=64); p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
REF_ONLY = a.run == "reference"
if REF_ONLY:
    cfg = dict(pitch="koma53", cond="prefix")
else:
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    vj = json.load(open(os.path.join(a.run, "vocab.json")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

corpus = D.load_corpus(("train", "val", "test"))
karar = D.karar_by_makam(corpus["train"]); ae2k = D.ae_to_koma(corpus["train"])
measure_len = D.measure_lengths(corpus["train"])
if cfg.get("smoke"): corpus = {k: v[::max(len(v) // 40, 1)] for k, v in corpus.items()}
D.tokenize_corpus(corpus, cfg["pitch"], karar, cfg["cond"] == "prefix", cfg.get("pos", False))
vocab = D.build_vocab(corpus["train"]) if REF_ONLY else Vocab(vj["itos"])
D.encode_corpus(corpus, vocab)
if not REF_ONLY:
    model = build_model(cfg["model"], len(vocab), cfg).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=dev)); model.eval()

# ---------- reference statistics from the training set
N = 8
pc_ref = collections.defaultdict(collections.Counter)      # makam -> pitch-class histogram
ngrams = set()
for x in corpus["train"]:
    k = karar[x["meta"]["makam"]]; ev = x["events"]
    seq = [(e["koma"], Fraction(e["num"], e["den"])) for e in ev]
    for e in ev:
        if e["koma"] != REST: pc_ref[x["meta"]["makam"]][(e["koma"] - k) % 53] += 1
    for i in range(len(seq) - N + 1): ngrams.add(tuple(seq[i:i + N]))
def norm(c):
    s = sum(c.values()) or 1; return {k: v / s for k, v in c.items()}
pc_ref = {m: norm(c) for m, c in pc_ref.items()}


def parse_groups(tokens, makam):
    """Token list -> list of (bar_flag, koma, Fraction) plus the number of malformed groups."""
    notes, bad, i, k = [], 0, 0, karar.get(makam)
    while i < len(tokens):
        bar = tokens[i] == BAR
        j = i + 1 if bar else i
        if j < len(tokens) and tokens[j].startswith("POS:"): j += 1
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
    return dict(pc=collections.defaultdict(collections.Counter), bars=[0, 0], ng=[0, 0], grp=[0, 0], off=[0, 0],
                karar=[0, 0], eos=[0, 0], length=[])

def update(st, notes, bad, makam, usul, ended=None):
    ref = pc_ref.get(makam, {}); k = karar.get(makam)
    for _, koma, _ in notes:
        if koma == REST: continue
        pc = (koma - k) % 53; st["pc"][makam][pc] += 1
        st["off"][1] += 1; st["off"][0] += ref.get(pc, 0.0) < 0.005
    st["grp"][0] += len(notes); st["grp"][1] += len(notes) + bad
    idx = [i for i, n in enumerate(notes) if n[0]]          # complete measures: between two BAR flags
    for s, e in zip(idx, idx[1:]):
        st["bars"][1] += 1
        st["bars"][0] += sum(n[2] for n in notes[s:e]) == measure_len.get(usul)
    seq = [(n[1], n[2]) for n in notes]
    for i in range(len(seq) - N + 1):
        st["ng"][1] += 1; st["ng"][0] += tuple(seq[i:i + N]) in ngrams
    if a.full:
        snd = [n[1] for n in notes if n[1] != REST]
        st["karar"][1] += 1; st["karar"][0] += bool(snd) and snd[-1] == k
        st["length"].append(len(notes))
        if ended is not None: st["eos"][1] += 1; st["eos"][0] += ended

def finish(st):
    ov, w = 0.0, 0
    for m, c in st["pc"].items():
        g = norm(c); n = sum(c.values())
        ov += n * sum(min(g.get(i, 0), pc_ref.get(m, {}).get(i, 0)) for i in range(53)); w += n
    r = lambda x: x[0] / x[1] if x[1] else None
    out = dict(scale_overlap=ov / w if w else None, off_scale_rate=r(st["off"]), bar_validity=r(st["bars"]),
               copy_rate=r(st["ng"]), grammar_ok=r(st["grp"]), notes=st["off"][1], measures=st["bars"][1])
    if a.full:
        out.update(ends_on_karar=r(st["karar"]), eos_rate=r(st["eos"]),
                   mean_notes=sum(st["length"]) / len(st["length"]) if st["length"] else None)
    return out


print(f"[{os.path.basename(a.run.rstrip('/'))}] prompt={a.prompt} full={a.full} constrain={a.constrain} device={dev}", flush=True)
gen, real = stats(), stats()
need = a.prompt + (8 if a.full else a.new_notes)
pieces = [x for x in corpus["test"] if len(x["groups"]) >= need]
if a.max_pieces: pieces = pieces[:a.max_pieces]
for x in pieces:
    m, u = x["meta"]["makam"], x["meta"]["usul"]
    ref_groups = x["groups"][a.prompt:] if a.full else x["groups"][a.prompt:a.prompt + a.new_notes]
    update(real, *parse_groups(sum(ref_groups, []), m), m, u)

if not REF_ONLY:
    for b in range(0, len(pieces), a.batch):
        chunk = pieces[b:b + a.batch]
        heads, L, run, pin = [], [], [], []
        for x in chunk:
            cut = x["note_starts"][a.prompt] if a.prompt < len(x["note_starts"]) else len(x["ids"]) - 1
            heads.append([vocab.bos] + x["prefix_ids"] + x["ids"][:cut]); pin.append(1 + len(x["prefix_ids"]))
            notes, _ = parse_groups(sum(x["groups"][:a.prompt], []), x["meta"]["makam"])
            r = Fraction(0)                                  # duration elapsed in the prompt's last measure
            for bar, _, d in notes: r = d if bar else r + d
            run.append(r); L.append(measure_len.get(x["meta"]["usul"]) if a.constrain else None)
        new, ended = generate_batch(model, heads, vocab, a.max_tokens if a.full else a.new_notes * 5,
                                    None if a.full else a.new_notes, a.temperature, a.top_p, L, run,
                                    progress=250 if a.full else 0, pinned=pin)
        for x, ids, e in zip(chunk, new, ended):
            m, u = x["meta"]["makam"], x["meta"]["usul"]
            update(gen, *parse_groups(vocab.decode(ids), m), m, u, e)
        print(f"{min(b + a.batch, len(pieces))}/{len(pieces)}", flush=True)

tag = f"gen_prompt{a.prompt}" + ("_full" if a.full else "") + ("_constrained" if a.constrain else "")
res = dict(run=os.path.basename(a.run.rstrip("/")), scenario=tag, pieces=len(pieces), prompt_notes=a.prompt,
           new_notes=None if a.full else a.new_notes, full=a.full, constrained=a.constrain,
           temperature=a.temperature, top_p=a.top_p,
           generated=None if REF_ONLY else finish(gen), reference=finish(real))
if not REF_ONLY:
    json.dump(res, open(os.path.join(a.run, tag + ".json"), "w"), indent=1)
print(json.dumps(res, indent=1))
