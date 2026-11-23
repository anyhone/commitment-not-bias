"""Evaluates one condition on a split.

examples:
  python scripts/evaluate.py --model llama --split dev --method baseline --limit 100
  python scripts/evaluate.py --model llama --split test --method caa --layer 14 --coef -4 --mmlu-sources bbq ss
  python scripts/evaluate.py --model llama --split test --method fairsteer --layer 16 --coef -4
  python scripts/evaluate.py --model llama --split test --method random --layer 14 --coef -4
"""
import argparse

from _common import get_logger, mdir, run_condition
from steering.model import load_config, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--split", choices=["dev", "test"], required=True)
ap.add_argument("--method", choices=["baseline", "caa", "fairsteer", "random"], required=True)
ap.add_argument("--layer", type=int)
ap.add_argument("--coef", type=float, default=0.0)
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--limit", type=int)
ap.add_argument("--mmlu-sources", nargs="+", default=["bbq"])
ap.add_argument("--raw", action="store_true", help="coefficient on the unnormalised DSV")
ap.add_argument("--vector", choices=["global", "category"], default="global")
ap.add_argument("--probe-set", default="probes", help="probes (code-specification labels) or probes_text (text-specification labels)")
a = ap.parse_args()

cfg = load_config(a.model)
log = get_logger(a.model, "evaluate")
model, tok = load_model(cfg)
tag = a.method if a.method == "baseline" else f"{a.method}_L{a.layer}_{'raw' if a.raw else 'rel'}{a.coef:g}"
if a.method == "fairsteer" and a.threshold != 0.5:
    tag += f"_t{a.threshold:g}"
if a.vector != "global" and a.method != "baseline":
    tag += "_cat"
if a.method == "fairsteer" and a.probe_set != "probes":
    tag += f"_{a.probe_set}"
out = mdir(a.model) / "eval" / (a.split + (f"_limit{a.limit}" if a.limit else "")) / tag
run_condition(model, tok, cfg, a.model, a.method, a.layer, a.coef, a.split, a.limit,
              tuple(a.mmlu_sources), a.threshold, out, log, raw=a.raw, vector=a.vector,
              probe_set=a.probe_set)
