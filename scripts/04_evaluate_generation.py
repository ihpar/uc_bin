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
  bar_validity    share of complete measures whose durations sum to the usul's usual measure length
  bar_validity_meter  the same against the measure length of the target piece itself
  copy_rate       share of 8-note (pitch, duration) n-grams that occur in the training set
  grammar_ok      share of token groups that follow the [BAR [REM]] [POS] [TIE] P D pattern
  --full only:
  ends_on_karar   share of pieces whose last sounded note is the karar of the makam
  eos_rate        share of pieces the model ended itself (before the token limit)
  mean_notes      mean piece length in notes
  length_ratio    generated measures / measures of the target piece (mean over pieces)
  repeated_measures  share of measures identical to one of the two measures before it
  pitch_by_tenth  mean pitch in commas above the karar for each tenth of the piece (the seyir)
  ending diagnostics: ends_on_karar_any_octave, karar_in_last_measure, final_note_dur (mean,
  whole notes), final_pitch_top (most frequent final pitches in commas above the karar)
  --save_samples writes <scenario>_samples.jsonl: the last three measures of every piece
  (generated and real) and the first 20 generated pieces in full.
"""
import argparse, collections, json, os, sys
from fractions import Fraction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src import dataset as D
from src.models import build_model
from src.generation import generate_batch
from src.symbtr import REST
from src.tokenizer import Vocab, decode_pitch, BAR, TIE

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("--prompt", type=int, default=16)
p.add_argument("--new_notes", type=int, default=64); p.add_argument("--max_pieces", type=int, default=0)
p.add_argument("--full", action="store_true", help="generate until EOS instead of --new_notes")
p.add_argument("--max_tokens", type=int, default=3000, help="token limit for --full")
p.add_argument("--constrain", action="store_true", help="usul-constrained decoding")
p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--batch", type=int, default=64); p.add_argument("--seed", type=int, default=1)
p.add_argument("--save_samples", action="store_true")
p.add_argument("--copy_penalty", type=float, default=0.0, help="logit penalty on literally repeating a recent measure")
p.add_argument("--suffix", default="", help="appended to the output file name")
a = p.parse_args()
REF_ONLY = a.run == "reference"
if REF_ONLY:
    cfg = dict(pitch="koma53", cond="prefix")
else:
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    vj = json.load(open(os.path.join(a.run, "vocab.json")))
DUR_FIRST = cfg.get("order") == "dp"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

corpus = D.load_corpus(("train", "val", "test"))
karar = D.karar_by_makam(corpus["train"]); ae2k = D.ae_to_koma(corpus["train"])
measure_len = D.measure_lengths(corpus["train"])
if cfg.get("smoke"): corpus = {k: v[::max(len(v) // 40, 1)] for k, v in corpus.items()}
D.tokenize_corpus(corpus, cfg["pitch"], karar, cfg["cond"] == "prefix", cfg.get("pos", False),
                  cfg.get("meter", False), cfg.get("rem", False), cfg.get("order") == "dp")
vocab = D.build_vocab(corpus["train"]) if REF_ONLY else Vocab(vj["itos"])
D.encode_corpus(corpus, vocab)
if not REF_ONLY:
    model = build_model(cfg["model"], len(vocab), cfg).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=dev)); model.eval()

def merge_ties(notes):
    """[(koma, duration, tie)] -> [(koma, duration)] with tied parts joined into one note,
    so that pitch and n-gram statistics count a held note once."""
    out = []
    for koma, d, tie in notes:
        if tie and out and out[-1][0] == koma: out[-1] = (koma, out[-1][1] + d)
        else: out.append((koma, d))
    return out


# ---------- reference statistics from the training set
N = 8
pc_ref = collections.defaultdict(collections.Counter)      # makam -> pitch-class histogram
ngrams = set()
for x in corpus["train"]:
    k = karar[x["meta"]["makam"]]; ev = x["events"]
    seq = merge_ties([(e["koma"], Fraction(e["num"], e["den"]), e["tie"]) for e in ev])
    for koma, _ in seq:
        if koma != REST: pc_ref[x["meta"]["makam"]][(koma - k) % 53] += 1
    for i in range(len(seq) - N + 1): ngrams.add(tuple(seq[i:i + N]))
def norm(c):
    s = sum(c.values()) or 1; return {k: v / s for k, v in c.items()}
pc_ref = {m: norm(c) for m, c in pc_ref.items()}


def parse_groups(tokens, makam):
    """Token list -> list of (bar_flag, koma, Fraction, tie) plus the number of malformed groups."""
    notes, bad, i, k = [], 0, 0, karar.get(makam)
    while i < len(tokens):
        bar = tokens[i] == BAR
        j = i + 1 if bar else i
        if bar and j < len(tokens) and tokens[j].startswith("REM:"): j += 1
        if j < len(tokens) and tokens[j].startswith("POS:"): j += 1
        tie = j < len(tokens) and tokens[j] == TIE
        if tie: j += 1
        a1, a2 = ("D:", "P:") if DUR_FIRST else ("P:", "D:")
        if j + 1 < len(tokens) and tokens[j].startswith(a1) and tokens[j + 1].startswith(a2):
            pt, dt = (tokens[j + 1], tokens[j]) if DUR_FIRST else (tokens[j], tokens[j + 1])
            koma = decode_pitch(pt, cfg["pitch"], k, ae2k)
            num, den = dt[2:].split("/")
            if koma is None: bad += 1
            else: notes.append((bar, koma, Fraction(int(num), int(den)), tie))
            i = j + 2
        else:
            bad += 1; i += 1
    return notes, bad


def stats():
    return dict(pc=collections.defaultdict(collections.Counter), bars=[0, 0], barsm=[0, 0], ng=[0, 0], grp=[0, 0],
                off=[0, 0], karar=[0, 0], eos=[0, 0], length=[], ratio=[],
                karar8=[0, 0], klast=[0, 0], fdur=[], fpitch=collections.Counter(), tails=[],
                rep=[], tenth=[[] for _ in range(10)])

def update(st, notes, bad, x, ended=None):
    makam, usul = x["meta"]["makam"], x["meta"]["usul"]
    ref = pc_ref.get(makam, {}); k = karar.get(makam)
    joined = merge_ties([n[1:] for n in notes])
    for koma, _ in joined:
        if koma == REST: continue
        pc = (koma - k) % 53; st["pc"][makam][pc] += 1
        st["off"][1] += 1; st["off"][0] += ref.get(pc, 0.0) < 0.005
    st["grp"][0] += len(notes); st["grp"][1] += len(notes) + bad
    idx = [i for i, n in enumerate(notes) if n[0]]          # complete measures: between two BAR flags
    for s, e in zip(idx, idx[1:]):
        tot = sum(n[2] for n in notes[s:e])
        st["bars"][1] += 1; st["bars"][0] += tot == measure_len.get(usul)
        st["barsm"][1] += 1; st["barsm"][0] += tot == x["meter"]
    for i in range(len(joined) - N + 1):
        st["ng"][1] += 1; st["ng"][0] += tuple(joined[i:i + N]) in ngrams
    if a.full:
        snd = [n[0] for n in joined if n[0] != REST]
        st["karar"][1] += 1; st["karar"][0] += bool(snd) and snd[-1] == k
        st["length"].append(len(joined))
        meas = [tuple((n[1], n[2]) for n in notes[s_:e_]) for s_, e_ in zip(idx, idx[1:] + [len(notes)])]
        if len(meas) > 2:
            st["rep"].append(sum(meas[i] in (meas[i - 1], meas[i - 2]) for i in range(2, len(meas))) / (len(meas) - 2))
        rel = [n[0] - k for n in joined if n[0] != REST]
        for d in range(10):
            seg = rel[len(rel) * d // 10:len(rel) * (d + 1) // 10]
            if seg: st["tenth"][d].append(sum(seg) / len(seg))
        # ending diagnostics
        last_bar = idx[-1] if idx else 0
        tail = [n for n in notes[last_bar:]]
        st["karar8"][1] += 1; st["karar8"][0] += bool(snd) and (snd[-1] - k) % 53 == 0
        st["klast"][1] += 1; st["klast"][0] += any(n[1] == k for n in tail)
        if snd:
            st["fpitch"][snd[-1] - k] += 1
            st["fdur"].append(float([n for n in joined if n[0] != REST][-1][1]))
        start = idx[-3] if len(idx) >= 3 else 0             # last three measures, relative to the karar
        st["tails"].append([["BAR" if n[0] else "", (n[1] - k) if n[1] != REST else "R", str(n[2]), int(n[3])]
                            for n in notes[start:]])
        if x["n_bars"]: st["ratio"].append(len(idx) / x["n_bars"])
        if ended is not None: st["eos"][1] += 1; st["eos"][0] += ended

def finish(st):
    ov, w = 0.0, 0
    for m, c in st["pc"].items():
        g = norm(c); n = sum(c.values())
        ov += n * sum(min(g.get(i, 0), pc_ref.get(m, {}).get(i, 0)) for i in range(53)); w += n
    r = lambda x: x[0] / x[1] if x[1] else None
    out = dict(scale_overlap=ov / w if w else None, off_scale_rate=r(st["off"]), bar_validity=r(st["bars"]),
               bar_validity_meter=r(st["barsm"]),
               copy_rate=r(st["ng"]), grammar_ok=r(st["grp"]), notes=st["off"][1], measures=st["bars"][1])
    if a.full:
        out.update(ends_on_karar=r(st["karar"]), eos_rate=r(st["eos"]),
                   mean_notes=sum(st["length"]) / len(st["length"]) if st["length"] else None,
                   length_ratio=sum(st["ratio"]) / len(st["ratio"]) if st["ratio"] else None,
                   ends_on_karar_any_octave=r(st["karar8"]), karar_in_last_measure=r(st["klast"]),
                   final_note_dur=sum(st["fdur"]) / len(st["fdur"]) if st["fdur"] else None,
                   final_pitch_top=[[p_, c] for p_, c in st["fpitch"].most_common(8)],
                   repeated_measures=sum(st["rep"]) / len(st["rep"]) if st["rep"] else None,
                   pitch_by_tenth=[round(sum(v) / len(v), 1) if v else None for v in st["tenth"]])
    return out


print(f"[{os.path.basename(a.run.rstrip('/'))}] prompt={a.prompt} full={a.full} constrain={a.constrain} device={dev}", flush=True)
gen, real = stats(), stats()
full_samples = []
need = a.prompt + (8 if a.full else a.new_notes)
pieces = [x for x in corpus["test"] if len(x["groups"]) >= need]
if a.max_pieces: pieces = pieces[:a.max_pieces]
for x in pieces:
    ref_groups = x["groups"][a.prompt:] if a.full else x["groups"][a.prompt:a.prompt + a.new_notes]
    update(real, *parse_groups(sum(ref_groups, []), x["meta"]["makam"]), x)

if not REF_ONLY:
    for b in range(0, len(pieces), a.batch):
        chunk = pieces[b:b + a.batch]
        heads, L, run, pin, rem = [], [], [], [], []
        use_meter = cfg.get("meter", False)
        for x in chunk:
            cut = x["note_starts"][a.prompt] if a.prompt < len(x["note_starts"]) else len(x["ids"]) - 1
            heads.append([vocab.bos] + x["prefix_ids"] + x["ids"][:cut]); pin.append(1 + len(x["prefix_ids"]))
            notes, _ = parse_groups(sum(x["groups"][:a.prompt], []), x["meta"]["makam"])
            r = Fraction(0)                                  # duration elapsed in the prompt's last measure
            for bar, _, d, _ in notes: r = d if bar else r + d
            run.append(r)
            L.append((x["meter"] if use_meter else measure_len.get(x["meta"]["usul"])) if a.constrain else None)
            started = sum(1 for n in notes if n[0])          # measures already started in the prompt
            rem.append(x["n_bars"] - started - 1 if cfg.get("rem") else None)
        new, ended = generate_batch(model, heads, vocab, a.max_tokens if a.full else a.new_notes * 5,
                                    None if a.full else a.new_notes, a.temperature, a.top_p, L, run,
                                    progress=250 if a.full else 0, pinned=pin, remaining=rem, dur_first=DUR_FIRST, copy_penalty=a.copy_penalty)
        for x, ids, e in zip(chunk, new, ended):
            toks = vocab.decode(ids)
            update(gen, *parse_groups(toks, x["meta"]["makam"]), x, e)
            if len(full_samples) < 20:
                full_samples.append(dict(fname=x["fname"], prefix=x["prefix"], ended=bool(e), tokens=" ".join(toks)))
        print(f"{min(b + a.batch, len(pieces))}/{len(pieces)}", flush=True)

tag = f"gen_prompt{a.prompt}" + ("_full" if a.full else "") + ("_constrained" if a.constrain else "") + a.suffix
res = dict(run=os.path.basename(a.run.rstrip("/")), scenario=tag, pieces=len(pieces), prompt_notes=a.prompt,
           new_notes=None if a.full else a.new_notes, full=a.full, constrained=a.constrain,
           temperature=a.temperature, top_p=a.top_p, copy_penalty=a.copy_penalty,
           generated=None if REF_ONLY else finish(gen), reference=finish(real))
if not REF_ONLY:
    json.dump(res, open(os.path.join(a.run, tag + ".json"), "w"), indent=1)
    if a.save_samples and a.full:
        with open(os.path.join(a.run, tag + "_samples.jsonl"), "w", encoding="utf-8") as fh:
            for x, g, t in zip(pieces, gen["tails"], real["tails"]):
                fh.write(json.dumps(dict(fname=x["fname"], makam=x["meta"]["makam"], usul=x["meta"]["usul"],
                                         meter=str(x["meter"]), generated_tail=g, real_tail=t), ensure_ascii=False) + "\n")
            for smp in full_samples:
                fh.write(json.dumps(dict(full=smp), ensure_ascii=False) + "\n")
print(json.dumps(res, indent=1))
