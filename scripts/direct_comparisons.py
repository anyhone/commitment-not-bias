"""Direct test-set comparisons between conditions (same items): FairSteer − CAA and CAA − random.

Usage: uv run python scripts/direct_comparisons.py <model> [<model> ...]
Output: results/<model>/report/direct_comparisons.json
Paired bootstrap as in report.py (1,000 resamples, clustered by BBQ template / StereoSet target).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steering.metrics import bbq_metrics, mcnemar, paired_bootstrap, ss_metrics  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def tag(meth, cfg):
    return f"{meth}_L{int(cfg['layer'])}_{cfg.get('coef_mode', 'rel')}{cfg['coef']:g}"


def compare(m, a, b, bench):
    T = RES / m / "eval" / "test"
    out = {"a": a, "b": b}
    if bench == "bbq":
        da, db = pd.read_parquet(T / a / "bbq.parquet"), pd.read_parquet(T / b / "bbq.parquet")
        for k in ["s_amb", "raw_s_amb", "acc_dis", "acc_amb"]:
            ci = paired_bootstrap(bbq_metrics, da, db, k, n=1000, cluster="template")
            out[k] = [ci["diff"], ci["ci_lo"], ci["ci_hi"]]
        m_dis = da.context_condition == "disambig"
        out["p_acc_dis"] = mcnemar((da.pred == da.label)[m_dis], (db.pred == db.label)[m_dis])["p"]
    else:
        da, db = pd.read_parquet(T / a / "ss.parquet"), pd.read_parquet(T / b / "ss.parquet")
        for k in ["SS", "LMS"]:
            ci = paired_bootstrap(ss_metrics, da, db, k, n=1000, cluster="target")
            out[k] = [ci["diff"], ci["ci_lo"], ci["ci_hi"]]
    return out


def main():
    for m in sys.argv[1:]:
        sel = json.load(open(RES / m / "report" / "selection.json"))
        res = {}
        for bench in ("bbq", "ss"):
            caa, fs = tag("caa", sel["caa"][bench]), tag("fairsteer", sel["fairsteer"][bench])
            rnd = "random" + caa[3:]
            res[f"fs_minus_caa_{bench}"] = compare(m, caa, fs, bench)   # b − a
            res[f"caa_minus_random_{bench}"] = compare(m, rnd, caa, bench)
        json.dump(res, open(RES / m / "report" / "direct_comparisons.json", "w"), indent=1)
        print("written", m, flush=True)


if __name__ == "__main__":
    main()
