"""Utilities shared by the scripts: paths, loading results, running one condition."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from steering import data as D  # noqa: E402
from steering.experiment import make_steerer, random_like, run_bbq, run_mmlu, run_ss, to_device  # noqa: E402
from steering.metrics import bbq_metrics, mmlu_metrics, ss_metrics  # noqa: E402

RES = V2 / "results"


def mdir(model: str) -> Path:
    p = RES / model
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(model: str, name: str) -> logging.Logger:
    (V2 / "logs").mkdir(exist_ok=True)
    lg = logging.getLogger(f"{model}.{name}")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s")
        for h in [logging.FileHandler(V2 / "logs" / f"{model}_{name}.log"), logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt); lg.addHandler(h)
    return lg


def datasets_for(split: str, limit: int | None = None):
    """Evaluation data per split. dev: BBQ subsample of 2,000 items (fixed seed), all of dev StereoSet,
    stratified MMLU validation. test: everything."""
    bbq, ss = D.with_splits(D.load_bbq(), D.load_stereoset())
    bbq = bbq[(bbq.split == split) & bbq.target_idx.notna()].reset_index(drop=True)
    ss = ss[ss.split == split].reset_index(drop=True)
    if split == "dev":
        bbq = bbq.sample(n=min(2000, len(bbq)), random_state=D.SEED).reset_index(drop=True)
        mmlu = D.mmlu_dev_subset()
    else:
        mmlu = D.load_mmlu("test")
    if limit:
        bbq, ss, mmlu = (x.sample(n=min(limit, len(x)), random_state=D.SEED).reset_index(drop=True)
                         for x in (bbq, ss, mmlu))
    return bbq, ss, mmlu


def load_vectors(model: str) -> dict:
    return torch.load(mdir(model) / "vectors.pt", weights_only=False)


def load_probe(model: str, layer: int, probe_set: str = "probes") -> dict:
    p = torch.load(mdir(model) / probe_set / f"probe_L{layer}.pt", weights_only=False)
    return {k: v.cuda() for k, v in p["torch"].items()}


def rel_scale(vecs: dict, resid_norm: float) -> dict:
    """Relative coefficient: each DSV becomes a unit direction x the mean residual-stream norm,
    so coef = −s means 'shift by a fraction s of the typical norm of h'."""
    return {k: v / v.norm().clamp_min(1e-8) * resid_norm for k, v in vecs.items()}


def run_condition(model_obj, tok, cfg, name: str, method: str, layer: int | None, coef: float,
                  split: str, limit: int | None = None, mmlu_sources=("bbq",), threshold=0.5,
                  out_dir: Path | None = None, log=None, raw: bool = False, vector: str = "global",
                  probe_set: str = "probes") -> dict:
    """Runs one condition on BBQ/StereoSet/MMLU of the split and saves per-item predictions + metrics.
    By default the coefficient is relative (see rel_scale); raw=True uses the unnormalised DSV.
    vector='global': a single DSV per dataset (mean over all pairs, as in FairSteer);
    vector='category': one DSV per category (ablation: assumes the category is known)."""
    bbq, ss, mmlu = datasets_for(split, limit)
    t0 = time.time()
    steerer, vb, vs, vm = None, None, None, {}
    if method != "baseline":
        V = load_vectors(name)
        vb, vs = dict(V["bbq"][layer]), dict(V["ss"][layer])
        if not raw:
            rn = V["meta"]["resid_norm"][layer]
            vb, vs = rel_scale(vb, rn), rel_scale(vs, rn)
        if method == "random":
            vb, vs = random_like(vb, seed=1), random_like(vs, seed=2)
        if vector == "global":
            vb = {c: vb["__pooled__"] for c in D.BBQ_CATS} | {"__pooled__": vb["__pooled__"]}
            vs = {c: vs["__pooled__"] for c in D.SS_CATS} | {"__pooled__": vs["__pooled__"]}
        elif vector != "category":
            raise ValueError(vector)
        vb, vs = to_device(vb), to_device(vs)
        vm = {"bbq": vb["__pooled__"], "ss": vs["__pooled__"]}  # MMLU: always the global vector
        probe = load_probe(name, layer, probe_set) if method == "fairsteer" else None
        steerer = make_steerer(model_obj, method, layer, coef, probe, threshold)
    try:
        rb, _ = run_bbq(model_obj, tok, cfg, bbq, steerer, vb, desc=f"{method}:bbq")
        rs, _ = run_ss(model_obj, tok, cfg, ss, steerer, vs, desc=f"{method}:ss")
        rm = {}
        for src in (mmlu_sources if method != "baseline" else ("none",)):
            rm[src], _ = run_mmlu(model_obj, tok, cfg, mmlu, steerer, vm.get(src), desc=f"{method}:mmlu-{src}")
    finally:
        if steerer is not None:
            steerer.remove()
    res = {"method": method, "layer": layer, "coef": coef, "coef_mode": "raw" if raw else "rel", "vector": vector, "probe_set": probe_set, "split": split, "limit": limit,
           "seconds": round(time.time() - t0),
           "bbq": bbq_metrics(rb), "ss": ss_metrics(rs),
           "mmlu": {k: mmlu_metrics(v) for k, v in rm.items()},
           "steer_rate": {"bbq": float(np.nanmean(rb.flag)) if rb.flag.notna().any() else None,
                          "ss": float(np.nanmean(rs.flag)) if rs.flag.notna().any() else None,
                          **{f"mmlu_{k}": (float(np.nanmean(v.flag)) if v.flag.notna().any() else None)
                             for k, v in rm.items()}}}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        rb.to_parquet(out_dir / "bbq.parquet"); rs.to_parquet(out_dir / "ss.parquet")
        for k, v in rm.items():
            v.to_parquet(out_dir / f"mmlu_{k}.parquet")
        (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    if log:
        log.info(f"{method} L{layer} {'raw' if raw else 'rel'}{coef} {vector} {split}: BBQ s_amb={res['bbq']['s_amb']:.4f} "
                 f"acc_amb={res['bbq']['acc_amb']:.2f} acc_dis={res['bbq']['acc_dis']:.2f} | "
                 f"SS={res['ss']['SS']:.2f} LMS={res['ss']['LMS']:.2f} | "
                 f"MMLU={ {k: round(v['acc'], 2) for k, v in res['mmlu'].items()} } | "
                 f"rate={res['steer_rate']} | {res['seconds']}s")
    return res
