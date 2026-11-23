"""Runs one condition (baseline / CAA / FairSteer / random vector) on BBQ, StereoSet and MMLU."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .hooks import Steerer
from .model import bbq_user_text, mmlu_user_text
from .scoring import ll_score, mc_score


def run_bbq(model, tok, cfg, df, steerer=None, vec_by_cat=None, capturer=None, desc="bbq"):
    vecs = None if vec_by_cat is None else [vec_by_cat[c] for c in df.category]
    r = mc_score(model, tok, cfg, [bbq_user_text(x) for _, x in df.iterrows()], 3, steerer, vecs, capturer,
                 batch_size=cfg["batch_size_mc"], desc=desc)
    out = df[["uid", "category", "template", "context_condition", "question_polarity", "label",
              "unknown_idx", "target_idx", "stereo_idx"]].copy()
    out["pred"] = r["pred"]
    for k in range(3):
        out[f"lp{k}"] = r["logprob"][:, k]
    out["flag"], out["prob_biased"] = r["flag"], r["prob_biased"]
    return out, r["acts"]


def run_ss(model, tok, cfg, df, steerer=None, vec_by_cat=None, capturer=None, capture_at="span", desc="ss"):
    out = df[["uid", "kind", "bias_type", "target"]].copy()
    acts = {}
    for opt in ["stereo", "anti", "unrelated"]:
        pairs = list(zip(df[f"ctx_{opt}"], df[f"comp_{opt}"]))
        vecs = None if vec_by_cat is None else [vec_by_cat[c] for c in df.bias_type]
        cap = capturer if (capturer is not None and opt != "unrelated") else None
        r = ll_score(model, tok, pairs, steerer, vecs, cap, batch_size=cfg["batch_size_ll"],
                     desc=f"{desc}:{opt}", capture_at=capture_at)
        out[f"ll_{opt}"] = r["ll"]
        if opt == "stereo":
            out["flag"] = r["flag"]
        if cap is not None:
            acts[opt] = r["acts"]
    return out, acts


def run_mmlu(model, tok, cfg, df, steerer=None, vec=None, capturer=None, desc="mmlu"):
    n_opts = len(df.choices.iloc[0])
    assert (df.choices.apply(len) == n_opts).all()
    r = mc_score(model, tok, cfg, [mmlu_user_text(x) for _, x in df.iterrows()], n_opts, steerer, vec, capturer,
                 batch_size=cfg["batch_size_mc"], desc=desc)
    out = df[["uid", "subject", "answer"]].copy()
    out["pred"] = r["pred"]
    out["flag"] = r["flag"]
    return out, r["acts"]


def make_steerer(model, method: str, layer: int, coef: float, probe=None, threshold=0.5):
    if method == "baseline":
        return None
    s = Steerer(model, layer)
    s.coef = coef
    if method in ("caa", "random"):
        s.mode = "caa"
    elif method == "fairsteer":
        s.mode, s.probe, s.threshold, s.last_only = "fairsteer", probe, threshold, True
    else:
        raise ValueError(method)
    return s


def random_like(vecs: dict, seed: int = 0) -> dict:
    """Random vectors with the same norm as each DSV (directional control)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in vecs.items():
        r = torch.randn(v.shape, generator=g)
        out[k] = r / r.norm() * v.norm()
    return out


def to_device(vecs: dict, device="cuda") -> dict:
    return {k: v.to(device=device, dtype=torch.float16) for k, v in vecs.items()}
