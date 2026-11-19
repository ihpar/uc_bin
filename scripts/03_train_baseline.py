"""Train a small language model on SymbTr and report held-out likelihood.

Examples:
  python scripts/03_train_baseline.py --model transformer --pitch koma53_rel
  python scripts/03_train_baseline.py --model lstm --pitch koma53_rel
  python scripts/03_train_baseline.py --smoke          # tiny CPU run to check the pipeline

Outputs in runs/<name>/: config.json, vocab.json, best.pt, metrics.json
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn.functional as F
from src import dataset as D
from src.models import build_model
from src.tokenizer import PITCH_MODES

ROOT = os.path.join(os.path.dirname(__file__), "..")
p = argparse.ArgumentParser()
p.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
p.add_argument("--pitch", choices=PITCH_MODES, default="koma53_rel")
p.add_argument("--cond", choices=["prefix", "none"], default="prefix", help="makam/usul/form prefix tokens")
p.add_argument("--pos", action="store_true", help="add POS tokens (duration elapsed in the current measure)")
p.add_argument("--meter", action="store_true", help="add a METER token (notated measure length) to the prefix")
p.add_argument("--rem", action="store_true", help="add REM tokens (measures remaining) after every BAR")
p.add_argument("--ctx", type=int, default=512)
p.add_argument("--d_model", type=int, default=256)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--lstm_layers", type=int, default=2)
p.add_argument("--lstm_hidden", type=int, default=512)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight_decay", type=float, default=0.01)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--patience", type=int, default=6, help="early stopping, in evaluations")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--name", default=None)
p.add_argument("--smoke", action="store_true")
p.add_argument("--print_name", action="store_true", help="print the run name and exit")
a = p.parse_args()
if a.smoke:
    a.ctx, a.d_model, a.layers, a.lstm_hidden, a.batch = 128, 64, 2, 128, 16
    a.steps, a.eval_every, a.warmup = 60, 30, 10
cfg = vars(a)
def auto_name():
    """Non-default hyperparameters are encoded in the run name so that sweeps do not collide."""
    n = f"{a.model}-{a.pitch}-{a.cond}" + ("-pos" if a.pos else "") + ("-meter" if a.meter else "") + ("-rem" if a.rem else "")
    if a.model == "transformer" and (a.layers, a.d_model) != (6, 256): n += f"-L{a.layers}d{a.d_model}"
    if a.model == "lstm" and (a.lstm_layers, a.lstm_hidden) != (2, 512): n += f"-L{a.lstm_layers}h{a.lstm_hidden}"
    if a.dropout != 0.1: n += f"-do{a.dropout}"
    if a.ctx != 512 and not a.smoke: n += f"-ctx{a.ctx}"
    return n + f"-s{a.seed}" + ("-smoke" if a.smoke else "")
name = a.name or auto_name()
if a.print_name:
    print(name); sys.exit(0)
cfg.pop("print_name")
out = os.path.join(ROOT, "runs", name); os.makedirs(out, exist_ok=True)
torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- data
corpus = D.load_corpus()
karar = D.karar_by_makam(corpus["train"], corpus["few_shot"])
if a.smoke:
    corpus = {k: v[::max(len(v) // 40, 1)] for k, v in corpus.items()}
D.tokenize_corpus(corpus, a.pitch, karar, a.cond == "prefix", a.pos, a.meter, a.rem)
vocab = D.build_vocab(corpus["train"])
D.encode_corpus(corpus, vocab)
n_tok = {k: sum(len(x["ids"]) for x in v) for k, v in corpus.items()}
print(f"[{name}] device={dev} vocab={len(vocab)} tokens={n_tok}")

model = build_model(a.model, len(vocab), cfg).to(dev)
n_params = sum(q.numel() for q in model.parameters())
decay = [q for n, q in model.named_parameters() if q.dim() >= 2]
no_decay = [q for n, q in model.named_parameters() if q.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                         {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
def lr_at(step):                      # linear warm-up, then cosine decay to 10% of the peak
    if step < a.warmup: return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(a.steps - a.warmup, 1)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else None


@torch.no_grad()
def evaluate(pieces):
    """Held-out negative log-likelihood. Because the encodings have different vocabularies,
    per-token perplexity is not comparable across them; bits per note is, with the caveat
    that tet12 predicts a coarser pitch target."""
    model.eval()
    nll = {"pitch": 0.0, "dur": 0.0, "bar": 0.0, "pos": 0.0, "rem": 0.0, "other": 0.0}; cnt = dict.fromkeys(nll, 0)
    unk = [0]
    kind = [vocab.kind(i) for i in range(len(vocab))]
    buf = []
    def flush():
        if not buf: return
        T = max(len(s[0]) for s in buf)
        X = torch.full((len(buf), T), vocab.pad, dtype=torch.long)
        for i, (inp, _, _) in enumerate(buf): X[i, :len(inp)] = torch.tensor(inp)
        logits = model(X.to(dev)).float().log_softmax(-1).cpu()
        for i, (inp, tgt, mask) in enumerate(buf):
            lp = logits[i, torch.arange(len(tgt)), torch.tensor(tgt)]
            for j, (t, m) in enumerate(zip(tgt, mask)):
                if m:
                    nll[kind[t]] -= lp[j].item(); cnt[kind[t]] += 1; unk[0] += t == vocab.unk
        buf.clear()
    for x in pieces:
        for w in D.eval_windows(x, vocab, a.ctx):
            buf.append(w)
            if len(buf) == 32: flush()
    flush()
    notes = max(sum(len(x["groups"]) for x in pieces), 1); tot = sum(nll.values()); ntok = max(sum(cnt.values()), 1)
    ln2 = math.log(2)
    return dict(nll_per_token=tot / ntok, ppl_per_token=math.exp(tot / ntok),
                bits_per_note=tot / notes / ln2, pitch_bits_per_note=nll["pitch"] / notes / ln2,
                dur_bits_per_note=nll["dur"] / notes / ln2, bar_bits_per_note=nll["bar"] / notes / ln2,
                pos_bits_per_note=nll["pos"] / notes / ln2, unk_rate=unk[0] / ntok, notes=notes, tokens=ntok)


# ---------- training
sampler = D.WindowSampler(corpus["train"], vocab, a.ctx, a.seed)
hist, best, bad, t0 = [], float("inf"), 0, time.time()
run_loss = 0.0
for step in range(a.steps):
    model.train()
    for g in opt.param_groups: g["lr"] = lr_at(step)
    X, Y = sampler.batch(a.batch, dev)
    if amp:
        with torch.autocast(**amp): logits = model(X)
    else:
        logits = model(X)
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    run_loss += loss.item()
    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        v = evaluate(corpus["val"])
        hist.append(dict(step=step + 1, train_loss=run_loss / a.eval_every, **v)); run_loss = 0.0
        print(f"step {step + 1:6d}  train {hist[-1]['train_loss']:.3f}  val bits/note {v['bits_per_note']:.3f}  "
              f"(pitch {v['pitch_bits_per_note']:.3f}, dur {v['dur_bits_per_note']:.3f})  {time.time() - t0:.0f}s", flush=True)
        if v["nll_per_token"] < best - 1e-4:
            best, bad = v["nll_per_token"], 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience: print("early stop"); break

model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
res = dict(name=name, config=cfg, device=dev, n_params=n_params, vocab_size=len(vocab), tokens=n_tok,
           train_seconds=round(time.time() - t0), steps_run=hist[-1]["step"],
           val=evaluate(corpus["val"]), test=evaluate(corpus["test"]),
           few_shot=evaluate(corpus["few_shot"]),          # makams never seen in training
           history=hist)
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
json.dump(dict(itos=vocab.itos, karar=karar), open(os.path.join(out, "vocab.json"), "w"), ensure_ascii=False)
json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("name", "n_params", "vocab_size", "steps_run", "train_seconds")}))
print("test:", json.dumps(res["test"]))
print("few_shot:", json.dumps(res["few_shot"]))
