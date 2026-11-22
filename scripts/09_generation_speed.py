"""How long does a composition request take?

    python scripts/09_generation_speed.py            # -> reports/results/generation_speed_<cpu|cuda>.json

Speed does not depend on what the weights are, only on the architecture, the
context length and the batch size, so this runs without the trained checkpoints:
the two models are built from their saved configuration and vocabulary with
random weights. It measures seconds per generated token with usul-constrained
decoding, starting from real test pieces, and combines that with the number of
tokens that real music needs per measure.

Requests timed:
  autocomplete   3 alternatives of 8 measures, continuing the first half of a piece
  from scratch   one piece of median length (Transformer), then 8 candidate endings
                 of 4 measures (LSTM), as in the best pipeline of the karar study
"""
from __future__ import annotations
import json, os, statistics, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import dataset as D
from src.models import build_model
from src.tokenizer import Vocab
from src.generation import generate_batch

RUNS = dict(transformer="transformer-koma53-prefix-pos-meter-rem-ctx768-s1",
            lstm="lstm-koma53-prefix-pos-meter-rem-do0.5-s1")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)

cfg = {k: json.load(open(os.path.join(ROOT, "runs", r, "config.json"))) for k, r in RUNS.items()}
vocab = Vocab(json.load(open(os.path.join(ROOT, "runs", RUNS["transformer"], "vocab.json")))["itos"])
c = cfg["transformer"]
corpus = D.load_corpus(("train", "test"))
karar = D.karar_by_makam(corpus["train"])
D.tokenize_corpus(corpus, c["pitch"], karar, True, True, True, True, False)
D.encode_corpus(corpus, vocab)
models = {k: build_model(cfg[k]["model"], len(vocab), cfg[k]).to(dev).eval() for k in cfg}

BAR = vocab.stoi["BAR"]
test = [x for x in corpus["test"] if x["n_bars"] >= 24]
tok_per_measure = [(len(x["ids"]) - x["ids"].index(BAR)) / x["n_bars"] for x in corpus["test"]]
piece_tokens = sorted(len(x["ids"]) for x in corpus["test"])
piece_bars = sorted(x["n_bars"] for x in corpus["test"])


def prompts(n):
    """First half of n test pieces, cut at a bar line: (head ids, pinned prefix, measure length, measures left)."""
    out = []
    for x in test[:: max(len(test) // n, 1)][:n]:
        ids = x["ids"]; bars = [i for i, t in enumerate(ids) if t == BAR]
        k = len(bars) // 2
        out.append((ids[:bars[k]], bars[0], x["meter"], x["n_bars"] - k))
    return out


def seconds_per_token(model, batch, n_prompts=3, new=32):
    """Median over prompts of the wall time per generated token (per step of the whole batch)."""
    ts = []
    for head, pin, L, left in prompts(n_prompts):
        t0 = time.perf_counter()
        out, _, _ = generate_batch(model, [head] * batch, vocab, new, None, measure_len=[L] * batch,
                                   running=[L] * batch, pinned=[pin] * batch, remaining=[left] * batch,
                                   copy_penalty=2.0)
        if dev == "cuda": torch.cuda.synchronize()
        steps = max(len(o) for o in out)
        ts.append((time.perf_counter() - t0) / max(steps, 1))
    return statistics.median(ts)


res = dict(device=dev, torch=torch.__version__, threads=torch.get_num_threads(),
           cpu=next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
           if os.path.exists("/proc/cpuinfo") else "?",
           gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
           tokens_per_measure=dict(median=statistics.median(tok_per_measure),
                                   p90=sorted(tok_per_measure)[int(0.9 * len(tok_per_measure))]),
           piece=dict(median_tokens=piece_tokens[len(piece_tokens) // 2], median_measures=piece_bars[len(piece_bars) // 2]),
           seconds_per_token={})
generate_batch(models["transformer"], [prompts(1)[0][0]], vocab, 4)          # warm-up
PREV = os.path.join(ROOT, "reports", "results", f"generation_speed_{dev}.json")
old = json.load(open(PREV)).get("seconds_per_token", {}) if os.environ.get("REUSE") and os.path.exists(PREV) else {}
for name, m in models.items():
    for b in (1, 3, 8):
        res["seconds_per_token"][f"{name}_batch{b}"] = old.get(f"{name}_batch{b}") or seconds_per_token(m, b)
        print(name, "batch", b, f"{1000 * res['seconds_per_token'][f'{name}_batch{b}']:.1f} ms per token", flush=True)

spt, tpm = res["seconds_per_token"], res["tokens_per_measure"]
res["requests"] = {
    "autocomplete_3x8_measures_transformer": dict(typical_s=8 * tpm["median"] * spt["transformer_batch3"],
                                                  dense_s=8 * tpm["p90"] * spt["transformer_batch3"]),
    "autocomplete_3x8_measures_lstm": dict(typical_s=8 * tpm["median"] * spt["lstm_batch3"],
                                           dense_s=8 * tpm["p90"] * spt["lstm_batch3"]),
    "from_scratch_median_piece": dict(
        body_s=res["piece"]["median_tokens"] * spt["transformer_batch1"],
        endings_s=4 * tpm["median"] * spt["lstm_batch8"]),
}
r = res["requests"]["from_scratch_median_piece"]; r["total_s"] = r["body_s"] + r["endings_s"]

# ---------------------------------------------------------------- cached decoding
# generate_batch re-runs the model over the whole context window for every new
# token. A service would keep the model state between tokens instead (hidden
# state for the LSTM, key/value cache for the Transformer). The step functions
# below do that with the same weights; they are checked against the full forward
# pass and timed at a long context. They are not yet used by generate_batch.
class Stub(torch.nn.Module):
    """Returns random logits at no cost: isolates the cost of sampling and constraints."""
    def __init__(self, n, ctx): super().__init__(); self.n, self.ctx = n, ctx; self.p = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x): return torch.randn(x.size(0), x.size(1), self.n)


@torch.no_grad()
def lstm_step(m, x, state):
    h, state = m.rnn(m.tok(x), state)
    return m.head(m.proj(h))[:, -1], state


@torch.no_grad()
def transformer_step(m, x, t, cache):
    """One new token x (B,1) at position t; cache[l] = (keys, values) of earlier positions."""
    h = m.tok(x) + m.pos(torch.tensor([t], device=x.device))
    for l, layer in enumerate(m.enc.layers):
        a = layer.self_attn; E = a.embed_dim; H = a.num_heads
        y = layer.norm1(h)
        q, k, v = torch.nn.functional.linear(y, a.in_proj_weight, a.in_proj_bias).split(E, -1)
        shape = lambda z: z.view(z.size(0), -1, H, E // H).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if cache[l] is not None:
            k = torch.cat([cache[l][0], k], 2); v = torch.cat([cache[l][1], v], 2)
        cache[l] = (k, v)
        att = torch.softmax(q @ k.transpose(-1, -2) / (E // H) ** 0.5, -1) @ v
        h = h + a.out_proj(att.transpose(1, 2).reshape(h.size(0), 1, E))
        h = h + layer.linear2(torch.nn.functional.gelu(layer.linear1(layer.norm2(h))))
    return m.head(m.norm(h))[:, -1]


@torch.no_grad()
def cached_times(batch, ctx_len=700, steps=40):
    x = torch.randint(5, len(vocab), (batch, ctx_len + steps), device=dev)
    out = {}
    # LSTM
    m = models["lstm"]; state = None
    _, state = m.rnn(m.tok(x[:, :ctx_len]))
    t0 = time.perf_counter()
    for t in range(ctx_len, ctx_len + steps): lg, state = lstm_step(m, x[:, t:t + 1], state)
    out["lstm_step_s"] = (time.perf_counter() - t0) / steps
    out["lstm_max_diff"] = float((lg - m(x)[:, -1]).abs().max())
    # Transformer
    m = models["transformer"]; cache = [None] * len(m.enc.layers)
    for t in range(ctx_len): transformer_step(m, x[:, t:t + 1], t, cache)
    t0 = time.perf_counter()
    for t in range(ctx_len, ctx_len + steps): lg = transformer_step(m, x[:, t:t + 1], t, cache)
    out["transformer_step_s"] = (time.perf_counter() - t0) / steps
    out["transformer_max_diff"] = float((lg - m(x)[:, -1]).abs().max())
    return out


res["cached"] = {f"batch{b}": cached_times(b) for b in (1, 3, 8)}
stub = Stub(len(vocab), 768)
res["sampling_overhead_s_per_token"] = {f"batch{b}": seconds_per_token(stub, b) for b in (1, 3, 8)}
for b in (1, 3, 8):
    print("cached, batch", b, {k: round(v * 1000, 2) if k.endswith("_s") else v for k, v in res["cached"][f"batch{b}"].items()},
          "overhead ms", round(1000 * res["sampling_overhead_s_per_token"][f"batch{b}"], 2), flush=True)
ch, ov = res["cached"], res["sampling_overhead_s_per_token"]
res["requests_with_cached_decoding"] = {
    # one full pass over the prompt, then cached steps
    "autocomplete_3x8_measures_transformer": dict(
        typical_s=spt["transformer_batch3"] + 8 * tpm["median"] * (ch["batch3"]["transformer_step_s"] + ov["batch3"]),
        dense_s=spt["transformer_batch3"] + 8 * tpm["p90"] * (ch["batch3"]["transformer_step_s"] + ov["batch3"])),
    "from_scratch_median_piece": dict(
        body_s=res["piece"]["median_tokens"] * (ch["batch1"]["transformer_step_s"] + ov["batch1"]),
        endings_s=spt["lstm_batch8"] + 4 * tpm["median"] * (ch["batch8"]["lstm_step_s"] + ov["batch8"])),
}
r = res["requests_with_cached_decoding"]["from_scratch_median_piece"]; r["total_s"] = r["body_s"] + r["endings_s"]

os.makedirs(os.path.join(ROOT, "reports", "results"), exist_ok=True)
out = os.path.join(ROOT, "reports", "results", f"generation_speed_{dev}.json")
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(dict(now=res["requests"], cached=res["requests_with_cached_decoding"]), indent=1)); print("wrote", out)
