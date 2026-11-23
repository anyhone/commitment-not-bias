"""Report: sweep table (dev), configuration selection, test-set comparisons with CIs.

Selection criterion (fixed before looking at the test set), separate per benchmark and method.
Candidates: all sweep conditions, with relative coefficient and with raw alpha (FairSteer reference);
the type (coef_mode) is stored in selection.json and passed to run_test.sh.
  BBQ      : minimum |s_amb| with acc_dis drop <= 2 points and MMLU drop <= 1 point w.r.t. the dev baseline.
  StereoSet: minimum |SS - 50| with LMS drop <= 2 points and MMLU drop <= 1 point.

usage: python scripts/report.py --model llama
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import mdir
from steering.metrics import bbq_metrics, mcnemar, mmlu_metrics, paired_bootstrap, ss_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
M = mdir(a.model)
OUT = M / "report"; OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- sweep dev
rows = []
for f in sorted((M / "eval" / "dev").glob("*/metrics.json")):
    r = json.loads(f.read_text())
    mm = r["mmlu"].get("bbq", r["mmlu"].get("none", {})).get("acc")
    rows.append({"tag": f.parent.name, "method": r["method"], "layer": r["layer"], "coef": r["coef"],
                 "coef_mode": r.get("coef_mode", "rel"),
                 "s_amb": r["bbq"]["s_amb"], "raw_s_amb": r["bbq"].get("raw_s_amb"), "s_dis": r["bbq"]["s_dis"], "acc_amb": r["bbq"]["acc_amb"],
                 "acc_dis": r["bbq"]["acc_dis"], "stereo_rate_amb": r["bbq"]["stereo_rate_amb"],
                 "SS": r["ss"]["SS"], "LMS": r["ss"]["LMS"], "ICAT": r["ss"]["ICAT"], "mmlu": mm,
                 "rate_bbq": r["steer_rate"]["bbq"], "rate_ss": r["steer_rate"]["ss"],
                 "rate_mmlu": r["steer_rate"].get("mmlu_bbq")})
sw = pd.DataFrame(rows)
sw.to_csv(OUT / "sweep_dev.csv", index=False)
base = sw[sw.method == "baseline"].iloc[0]

sel = {}
for meth in ["caa", "fairsteer"]:
    # candidates: sweep conditions only (runs with a non-default threshold, e.g. the post-hoc ungated `_t-1`, are excluded)
    s = sw[(sw.method == meth) & ~sw.tag.str.contains(r"_t-?\d", regex=True)]
    ok_b = s[(s.acc_dis >= base.acc_dis - 2) & (s.mmlu >= base.mmlu - 1)]
    ok_s = s[(s.LMS >= base.LMS - 2) & (s.mmlu >= base.mmlu - 1)]
    sel[meth] = {
        "bbq": ok_b.loc[ok_b.s_amb.abs().idxmin(), ["layer", "coef", "coef_mode"]].to_dict() if len(ok_b) else None,
        "ss": ok_s.loc[(ok_s.SS - 50).abs().idxmin(), ["layer", "coef", "coef_mode"]].to_dict() if len(ok_s) else None,
    }
(OUT / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
print("selection on dev:", json.dumps(sel, default=float))

# ---------------------------------------------------------------- sweep figures
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
for meth in ["caa", "fairsteer"]:
  caa = sw[(sw.method == meth) & (sw.coef_mode == "rel")]
  if len(caa):
      panels = [("s_amb", "BBQ bias score (ambiguous)"), ("SS", "StereoSet SS"), ("mmlu", "MMLU acc (%)"),
                ("acc_dis", "BBQ acc disambiguated (%)")]
      fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
      layers = sorted(caa.layer.unique())
      for ax, (k, title) in zip(axs, panels):
          for i, l in enumerate(layers):
              d = caa[caa.layer == l].sort_values("coef")
              ax.plot(d.coef, d[k], color=RAMP[min(i, len(RAMP) - 1)], lw=2, marker="o", ms=4, label=f"L{int(l)}")
          ax.axhline(base[k], color="#8a8a86", lw=1.5, ls="--", label="baseline")
          ax.set_title(title, fontsize=10); ax.set_xlabel("coefficient"); ax.grid(alpha=.25)
          for s_ in ["top", "right"]:
              ax.spines[s_].set_visible(False)
      axs[-1].legend(fontsize=7, frameon=False, ncol=2)
      fig.suptitle(f"{a.model}: {meth} sweep on dev (layer = colour, darker = deeper)", fontsize=11)
      fig.tight_layout(); fig.savefig(OUT / f"sweep_{meth}_dev.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- test-set comparisons
def amb_composition(d: pd.DataFrame) -> dict:
    """Answers on ambiguous contexts (%): unknown / stereotyped / anti-stereotyped.
    "Stereotyped" with the official definition of bbq_metrics (target if the question is negative, the other
    group if non-negative): shows whether s_amb drops through abstention or because answers change direction.
    bbq_amb_biased equals stereo_rate_amb; it is repeated so that the three columns sum to 100."""
    a = d[d.target_idx.notna() & (d.context_condition == "ambig")]
    unk = a.pred == a.unknown_idx
    neg = a.question_polarity == "neg"
    tgt = a.pred == a.target_idx
    biased = ~unk & ((neg & tgt) | (~neg & ~tgt))
    return {"bbq_amb_unknown": float(unk.mean() * 100), "bbq_amb_biased": float(biased.mean() * 100),
            "bbq_amb_anti": float((~unk & ~biased).mean() * 100)}


T = M / "eval" / "test"
if (T / "baseline").exists():
    b_bbq = pd.read_parquet(T / "baseline" / "bbq.parquet")
    b_ss = pd.read_parquet(T / "baseline" / "ss.parquet")
    b_mm = pd.read_parquet(T / "baseline" / "mmlu_none.parquet")
    res = [{"cond": "baseline", **{f"bbq_{k}": v for k, v in bbq_metrics(b_bbq).items()}, **amb_composition(b_bbq),
            **{f"ss_{k}": v for k, v in ss_metrics(b_ss).items()}, "mmlu_acc": mmlu_metrics(b_mm)["acc"]}]
    for d in sorted(p for p in T.iterdir() if p.is_dir() and p.name != "baseline"):
        x_bbq, x_ss = pd.read_parquet(d / "bbq.parquet"), pd.read_parquet(d / "ss.parquet")
        row = {"cond": d.name, **{f"bbq_{k}": v for k, v in bbq_metrics(x_bbq).items()}, **amb_composition(x_bbq),
               **{f"ss_{k}": v for k, v in ss_metrics(x_ss).items()}}
        for k, fn, key, cl in [("s_amb", bbq_metrics, "s_amb", "template"), ("raw_s_amb", bbq_metrics, "raw_s_amb", "template"),
                               ("acc_dis", bbq_metrics, "acc_dis", "template"),
                               ("SS", ss_metrics, "SS", "target"), ("LMS", ss_metrics, "LMS", "target")]:
            da, db = (b_bbq, x_bbq) if fn is bbq_metrics else (b_ss, x_ss)
            ci = paired_bootstrap(fn, da, db, key, n=1000, cluster=cl)
            row[f"d_{k}"], row[f"d_{k}_lo"], row[f"d_{k}_hi"] = ci["diff"], ci["ci_lo"], ci["ci_hi"]
        dis = b_bbq.context_condition == "disambig"
        row["p_acc_dis"] = mcnemar((b_bbq.pred == b_bbq.label)[dis], (x_bbq.pred == x_bbq.label)[dis])["p"]
        row["p_SS"] = mcnemar(b_ss.ll_stereo > b_ss.ll_anti, x_ss.ll_stereo > x_ss.ll_anti)["p"]
        for src in ["bbq", "ss"]:
            f = d / f"mmlu_{src}.parquet"
            if f.exists():
                x_mm = pd.read_parquet(f)
                row[f"mmlu_{src}_acc"] = mmlu_metrics(x_mm)["acc"]
                row[f"p_mmlu_{src}"] = mcnemar(b_mm.pred == b_mm.answer, x_mm.pred == x_mm.answer)["p"]
                row[f"rate_mmlu_{src}"] = float(np.nanmean(x_mm.flag)) if x_mm.flag.notna().any() else None
        row["rate_bbq"] = float(np.nanmean(x_bbq.flag)) if x_bbq.flag.notna().any() else None
        row["rate_ss"] = float(np.nanmean(x_ss.flag)) if x_ss.flag.notna().any() else None
        res.append(row)
    pd.DataFrame(res).to_csv(OUT / "test_results.csv", index=False)
    print(pd.DataFrame(res).round(4).T.to_string())
