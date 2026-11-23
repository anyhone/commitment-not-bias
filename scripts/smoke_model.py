"""Smoke test of a model before the full run.

Checks: number of layers and layer output type (tuple/tensor), end of the chat template,
probability mass on the letter tokens and preferred tokens, distribution of the predicted letters,
effect of the steering hook, peak GPU memory with the config batch sizes.

usage: python scripts/smoke_model.py --model qwen
"""
import argparse
import collections

import numpy as np
import torch

from _common import D
from steering.hooks import Steerer
from steering.model import bbq_user_text, chat_prompt, get_layers, letter_token_ids, load_config, load_model
from steering.scoring import ll_score, mc_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()
cfg = load_config(a.model)
model, tok = load_model(cfg)
ok = True


def check(cond, msg):
    global ok
    print(("OK   " if cond else "FAIL ") + msg)
    ok &= bool(cond)


layers = get_layers(model)
check(len(layers) == 32, f"layers found: {len(layers)}")
tc = model.config.get_text_config()
print("     layer_types:", collections.Counter(getattr(tc, "layer_types", None) or ["standard"]))
seen = {}
def _inspect(m, i, o):
    seen.setdefault("type", type(o).__name__)  # no return: the output must not be modified


h = layers[16].register_forward_hook(_inspect)
s, _ = chat_prompt(tok, cfg, "hello")
print("     template end:", repr(s[-60:]))
if cfg.get("template_kwargs", {}).get("enable_thinking") is False:
    check(s.endswith("<think>\n\n</think>\n\n"), "thinking disabled in the template")

b, ss = D.with_splits(D.load_bbq(), D.load_stereoset())
b = b[(b.split == "dev") & b.target_idx.notna()].sample(a.n, random_state=0).reset_index(drop=True)
ids = letter_token_ids(tok, 3)
allids = sorted({i for x in ids for i in x})
print("     letter tokens:", [tok.convert_ids_to_tokens(x) for x in ids])
mass, top = [], collections.Counter()
for _, r in b.iterrows():
    s, _ = chat_prompt(tok, cfg, bbq_user_text(r))
    x = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        p = torch.softmax(model(x).logits[0, -1].float(), -1)
    mass.append(p[allids].sum().item())
    top[tok.convert_ids_to_tokens(int(p.argmax()))] += 1
h.remove()
print("     layer output:", seen.get("type"))
check(np.mean(mass) > 0.8, f"mean mass on the letter tokens {np.mean(mass):.3f} (min {np.min(mass):.3f})")
print("     most likely tokens:", top.most_common(6))

torch.cuda.reset_peak_memory_stats()
texts = [bbq_user_text(r) for _, r in b.iterrows()]
r0 = mc_score(model, tok, cfg, texts, 3, batch_size=cfg["batch_size_mc"])
acc = (r0["pred"] == b.label.values).mean() * 100
print(f"     predicted letters: {collections.Counter(r0['pred'].tolist())} | acc {acc:.1f}% on {a.n} dev items")
check(acc > 45, "BBQ accuracy above chance (33%)")
st = Steerer(model, 16)
st.mode, st.coef = "caa", -1.0
v = torch.randn(tc.hidden_size, generator=torch.Generator().manual_seed(0))
v = (v / v.norm() * 10).to("cuda", torch.float16)
r1 = mc_score(model, tok, cfg, texts, 3, steerer=st, vecs=v, batch_size=cfg["batch_size_mc"])
st.remove()
check(np.abs(r1["logprob"] - r0["logprob"]).max() > 0.05, "steering changes the log-probabilities")
pairs = list(zip(ss.ctx_stereo.head(32), ss.comp_stereo.head(32)))
ll_score(model, tok, pairs, batch_size=cfg["batch_size_ll"])
peak = torch.cuda.max_memory_allocated() / 2**30
check(peak < 15.0, f"peak GPU memory {peak:.1f} GiB with batch mc={cfg['batch_size_mc']} ll={cfg['batch_size_ll']}")
print("SMOKE", "PASSED" if ok else "FAILED")
