"""Trains the BAD classifier for each layer from the baseline activations.

Labels (1 = biased), code specification (default, `--labels code`, as in the official FairSteer code):
  BBQ      : 1 if the model picks the stereotyped answer and it is wrong, 0 if it is correct; the rest excluded.
  StereoSet: 1 if ll(stereo) > ll(anti) at baseline, else 0 (activation: last context token).
  MMLU     : 0 (auxiliary_train, never used for evaluation).
Text specification (`--labels text`, a reading of Sec. 3.1 of the FairSteer paper): BBQ 1 if the model picks
the stereotyped answer, 0 if it picks "unknown".
Train = train split (+75% of the MMLU neutral examples); dev = dev split (+25% of them).
Selected layer = highest mean balanced accuracy on dev BBQ and StereoSet.

usage: python scripts/train_probe.py --model llama
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from _common import get_logger, mdir
from steering.model import load_config
from steering.probe import balanced_acc, fit_probe, to_torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--baseline", default="baseline")
ap.add_argument("--labels", choices=["code", "text"], default="code",
                help="code (code specification): biased = wrong stereotyped answer, unbiased = correct; "
                     "text (text specification, FairSteer Sec. 3.1): biased = stereotyped answer, unbiased = neutral answer")
a = ap.parse_args()
cfg = load_config(a.model)
log = get_logger(a.model, "probe")
B = mdir(a.model) / a.baseline
acts = B / "acts"

bbq = pd.read_parquet(B / "bbq.parquet")
ss = pd.read_parquet(B / "ss.parquet")
aux = pd.read_parquet(B / "mmlu_aux.parquet")
assert (np.load(acts / "bbq_uids.npy", allow_pickle=True) == bbq.uid.values).all()
assert (np.load(acts / "ss_uids.npy", allow_pickle=True) == ss.uid.values).all()

stereo = bbq.stereo_idx.fillna(-1).astype(int).values
pred, label = bbq.pred.values, bbq.label.values
if a.labels == "code":
    bbq["y"] = np.where((pred == stereo) & (pred != label), 1, np.where(pred == label, 0, -1))
else:
    unk = bbq.unknown_idx.values
    bbq["y"] = np.where(pred == stereo, 1, np.where(pred == unk, 0, -1))
bbq.loc[bbq.target_idx.isna().values, "y"] = -1
ss["y"] = (ss.ll_stereo > ss.ll_anti).astype(int)
aux["split"] = np.where(np.arange(len(aux)) < int(0.75 * len(aux)), "train", "dev")
aux["y"] = 0

PDIR = "probes" if a.labels == "code" else "probes_text"
rows = {}
best = None
for l in cfg["layers"]:
    Xb, Xs, Xa = (np.load(acts / f"{t}_L{l}.npy").astype(np.float32) for t in ["bbq", "ss", "mmlu_aux"])
    parts = {"bbq": (Xb, bbq), "ss": (Xs, ss), "mmlu": (Xa, aux)}

    def take(split):
        X, y, src = [], [], []
        for s, (Xs_, df) in parts.items():
            m = ((df.split == split) & (df.y >= 0)).values
            X.append(Xs_[m]); y.append(df.y.values[m]); src += [s] * int(m.sum())
        return np.vstack(X), np.concatenate(y), np.array(src)

    Xtr, ytr, str_ = take("train")
    Xdv, ydv, sdv = take("dev")
    p = fit_probe(Xtr, ytr)
    yh = p.predict(Xdv)
    r = {"n_train": int(len(ytr)), "n_dev": int(len(ydv)), "frac_biased_train": float(ytr.mean()),
         "acc_dev": float((yh == ydv).mean() * 100), "bacc_dev": balanced_acc(ydv, yh)}
    for s in ["bbq", "ss", "mmlu"]:
        m = sdv == s
        r[f"acc_{s}"] = float((yh[m] == ydv[m]).mean() * 100)
        r[f"bacc_{s}"] = balanced_acc(ydv[m], yh[m]) if s != "mmlu" else None
        r[f"rate_{s}"] = float(yh[m].mean() * 100)  # share predicted biased = expected steering rate
    # trivial baseline: majority class per source (how much the format alone explains)
    maj = {s: int(round(ytr[str_ == s].mean())) for s in ["bbq", "ss", "mmlu"]}
    r["acc_trivial_source"] = float(np.mean([maj[s] == y for s, y in zip(sdv, ydv)]) * 100)
    score = np.mean([r["bacc_bbq"], r["bacc_ss"]])
    r["select_score"] = float(score)
    rows[l] = r
    (mdir(a.model) / PDIR).mkdir(exist_ok=True)
    torch.save({"torch": {k: v.cpu() for k, v in to_torch(p, "cpu").items()}, "report": r},
               mdir(a.model) / PDIR / f"probe_L{l}.pt")
    log.info(f"L{l}: {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})}")
    if best is None or score > rows[best]["select_score"]:
        best = l

(mdir(a.model) / PDIR / "report.json").write_text(json.dumps({"best_layer": best, "layers": rows}, indent=1))
log.info(f"layer selected for FairSteer: {best}")
