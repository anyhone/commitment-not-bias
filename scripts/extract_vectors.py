"""DSV extraction from the `train` split (never used for evaluation).

BBQ  : ambiguous contexts; evaluation prompt + assistant turn with the letter of the stereotyped
       answer vs the letter of "unknown"; activation at the letter token.
       DSV = mean(stereo) − mean(unknown)  (points towards the bias).
StereoSet: context + stereotyped vs anti-stereotyped completion; mean over the completion tokens.
       DSV = mean(stereo) − mean(anti).
For each layer: one vector per category, '__all__' (mean of the category vectors) and '__pooled__'
(mean over all pairs: FairSteer's global DSV, the main condition).

usage: python scripts/extract_vectors.py --model llama
"""
import argparse

import numpy as np
import torch

from _common import D, get_logger, mdir
from steering.hooks import Capturer
from steering.model import LETTERS, bbq_user_text, load_config, load_model
from steering.scoring import ll_score, mc_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
cfg = load_config(a.model)
log = get_logger(a.model, "vectors")
model, tok = load_model(cfg)
layers = cfg["layers"]
cap = Capturer(model, layers)

bbq, ss = D.with_splits(D.load_bbq(), D.load_stereoset())
tr = bbq[(bbq.split == "train") & (bbq.context_condition == "ambig") & bbq.stereo_idx.notna()].reset_index(drop=True)
texts = [bbq_user_text(r) for _, r in tr.iterrows()]
acts = {}
for kind, col in [("stereo", "stereo_idx"), ("unknown", "unknown_idx")]:
    pref = [LETTERS[int(i)] for i in tr[col]]
    acts[kind] = mc_score(model, tok, cfg, texts, 3, capturer=cap, prefixes=pref,
                          batch_size=cfg["batch_size_mc"], desc=f"vec:bbq:{kind}")["acts"]

V = {"bbq": {}, "ss": {}, "meta": {"bbq_n": tr.groupby("category").size().to_dict()}}
for l in layers:
    V["bbq"][l] = {}
    for c in D.BBQ_CATS:
        m = (tr.category == c).values
        V["bbq"][l][c] = (acts["stereo"][l][m].float().mean(0) - acts["unknown"][l][m].float().mean(0))
    V["bbq"][l]["__all__"] = torch.stack([V["bbq"][l][c] for c in D.BBQ_CATS]).mean(0)
    # global as in FairSteer (Eq. 3): mean of the differences over all pairs of all categories
    V["bbq"][l]["__pooled__"] = acts["stereo"][l].float().mean(0) - acts["unknown"][l].float().mean(0)

st = ss[ss.split == "train"].reset_index(drop=True)
sacts = {}
for opt in ["stereo", "anti"]:
    sacts[opt] = ll_score(model, tok, list(zip(st[f"ctx_{opt}"], st[f"comp_{opt}"])), capturer=cap,
                          batch_size=cfg["batch_size_ll"], desc=f"vec:ss:{opt}", capture_at="span")["acts"]
V["meta"]["ss_n"] = st.groupby("bias_type").size().to_dict()
for l in layers:
    V["ss"][l] = {}
    for c in D.SS_CATS:
        m = (st.bias_type == c).values
        V["ss"][l][c] = sacts["stereo"][l][m].float().mean(0) - sacts["anti"][l][m].float().mean(0)
    V["ss"][l]["__all__"] = torch.stack([V["ss"][l][c] for c in D.SS_CATS]).mean(0)
    V["ss"][l]["__pooled__"] = sacts["stereo"][l].float().mean(0) - sacts["anti"][l].float().mean(0)

# mean residual-stream norm (reference for interpreting the coefficient)
V["meta"]["resid_norm"] = {l: float(acts["unknown"][l].float().norm(dim=1).mean()) for l in layers}
cap.remove()
torch.save(V, mdir(a.model) / "vectors.pt")
for l in layers:
    log.info(f"L{l}: |h|={V['meta']['resid_norm'][l]:.1f} | BBQ |DSV| " +
             " ".join(f"{c[:6]}={V['bbq'][l][c].norm():.2f}" for c in D.BBQ_CATS) +
             " | SS |DSV| " + " ".join(f"{c[:6]}={V['ss'][l][c].norm():.2f}" for c in D.SS_CATS))
log.info(f"vectors saved, n pairs: {V['meta']}")
