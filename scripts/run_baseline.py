"""Baseline on all splits + capture of the activations (last token of the prompt / of the context)
at the config layers, used by the BAD classifier.

usage: python scripts/run_baseline.py --model llama [--limit 200]
"""
import argparse

import numpy as np

from _common import D, get_logger, mdir, bbq_metrics, ss_metrics, mmlu_metrics
from steering.hooks import Capturer
from steering.experiment import run_bbq, run_mmlu, run_ss
from steering.model import load_config, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--limit", type=int, default=None, help="max items per dataset (smoke test)")
ap.add_argument("--only", nargs="+", default=["bbq", "ss", "mmlu_aux", "mmlu_eval"],
                help="parts to run (to redo only some of them)")
a = ap.parse_args()

cfg = load_config(a.model)
log = get_logger(a.model, "baseline")
model, tok = load_model(cfg)
out = mdir(a.model) / ("baseline" if not a.limit else f"baseline_limit{a.limit}")
acts_dir = out / "acts"; acts_dir.mkdir(parents=True, exist_ok=True)
layers = cfg["layers"]

bbq, ss = D.with_splits(D.load_bbq(), D.load_stereoset())
mmlu_test, mmlu_dev, mmlu_aux = D.load_mmlu("test"), D.mmlu_dev_subset(), D.mmlu_neutral_pool()
if a.limit:
    f = lambda x: x.sample(n=min(a.limit, len(x)), random_state=D.SEED).reset_index(drop=True)
    bbq, ss, mmlu_test, mmlu_dev, mmlu_aux = map(f, (bbq, ss, mmlu_test, mmlu_dev, mmlu_aux))


def save_acts(tag, uids, acts):
    np.save(acts_dir / f"{tag}_uids.npy", np.asarray(uids))
    for l, A in acts.items():
        np.save(acts_dir / f"{tag}_L{l}.npy", A.numpy())


cap = Capturer(model, layers)
if "bbq" in a.only:
  rb, ab = run_bbq(model, tok, cfg, bbq, capturer=cap, desc="base:bbq")
  rb["split"] = bbq.split.values
  rb.to_parquet(out / "bbq.parquet"); save_acts("bbq", rb.uid, ab)
  for sp in ["train", "dev", "test"]:
      log.info(f"BBQ {sp}: {bbq_metrics(rb[rb.split == sp])}")

if "ss" in a.only:
  rs, as_ = run_ss(model, tok, cfg, ss, capturer=cap, capture_at="cls", desc="base:ss")
  rs["split"] = ss.split.values
  rs.to_parquet(out / "ss.parquet"); save_acts("ss", rs.uid, as_["stereo"])
  for sp in ["train", "dev", "test"]:
      log.info(f"SS {sp}: {ss_metrics(rs[rs.split == sp])}")

if "mmlu_aux" in a.only:
  ra, aa = run_mmlu(model, tok, cfg, mmlu_aux, capturer=cap, desc="base:mmlu-aux")
  ra.to_parquet(out / "mmlu_aux.parquet"); save_acts("mmlu_aux", ra.uid, aa)
cap.remove()

if "mmlu_eval" in a.only:
  for tag, df in [("mmlu_dev", mmlu_dev), ("mmlu_test", mmlu_test)]:
      r, _ = run_mmlu(model, tok, cfg, df, desc=f"base:{tag}")
      r.to_parquet(out / f"{tag}.parquet")
      log.info(f"{tag}: {mmlu_metrics(r)}")
log.info("baseline done")
