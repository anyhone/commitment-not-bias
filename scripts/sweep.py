"""Layer and coefficient sweep on the dev split (the test split is never touched here).

CAA and FairSteer: same layers from the config x same coefficients; for FairSteer each layer uses its own
classifier. The layer of both methods is then chosen with the same criterion on the effect (report.py),
not on the probe accuracy (a deep layer detects well but leaves few layers to propagate the intervention).
Resumes where it stopped (conditions already present are skipped).

usage: python scripts/sweep.py --model llama [--coefs -0.05 -0.1 -0.2 -0.4 -0.8]  (coefficient relative to |h|) [--methods caa fairsteer]
"""
import argparse

from _common import get_logger, mdir, run_condition
from steering.model import load_config, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--coefs", nargs="+", type=float, default=[-0.02, -0.05, -0.1, -0.15, -0.2, -0.3, -0.5, -0.8])
ap.add_argument("--methods", nargs="+", default=["caa", "fairsteer"])
ap.add_argument("--layers", nargs="+", type=int)
ap.add_argument("--raw-coefs", nargs="+", type=float, default=[-1.0],
                help="coefficients on the raw DSV; -1 = alpha 1 of the FairSteer paper (reference)")
a = ap.parse_args()

cfg = load_config(a.model)
log = get_logger(a.model, "sweep")
model, tok = load_model(cfg)
root = mdir(a.model) / "eval" / "dev"
todo = []
if not (root / "baseline" / "metrics.json").exists():
    todo.append(("baseline", None, 0.0, False))
for l in (a.layers or cfg["layers"]):       # outer loop on layers: if the sweep stops, both methods
    for m in a.methods:                      # are covered on the same layers
        for c in a.coefs:
            todo.append((m, l, c, False))
        for c in a.raw_coefs:
            todo.append((m, l, c, True))
log.info(f"{len(todo)} conditions to run")
for m, l, c, raw in todo:
    tag = m if m == "baseline" else f"{m}_L{l}_{'raw' if raw else 'rel'}{c:g}"
    if (root / tag / "metrics.json").exists():
        continue
    run_condition(model, tok, cfg, a.model, m, l, c, "dev", out_dir=root / tag, log=log, raw=raw)
log.info("sweep done")
