"""Mechanism of the monitor. Post hoc and descriptive.

For each model and split (test primary, dev as a control), at FairSteer's BBQ configuration, on ambiguous items
grouped by baseline answer (S stereotyped, A anti-stereotyped, U unknown):
  1. selection (monitor activation rate, code and text specification) x conversion (ungated run,
     flagged and unflagged items) and decomposition of Δ raw_s_amb under the code gate;
  2. AUROC of the code probe between S and A (non-unknown ambiguous items) and mean score per group;
  3. oracles at k_amb (non-unknown, context) and the "all S" gate, compared with the monitor on ambiguous-item metrics;
  4. share of interventions wasted on ambiguous items already answered "unknown" at baseline.
Uses saved outputs only. 95% bootstrap CIs clustered by template (1,000 replicates).

Usage: uv run python scripts/posthoc_mechanism.py [model ...]
Output: results/<model>/report/posthoc_mechanism.json, log logs/posthoc_mechanism.log
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_gates import BBQ, RES, V2, boot_weights, confidence, load, probe_scores, tag  # noqa: E402

N_BOOT, N_DRAWS = 1000, 1000
AMB_METRICS = ["s_amb", "raw_s_amb", "unk_amb"]
GROUPS = ["S", "A", "U"]

log = logging.getLogger("posthoc_mech")
log.setLevel(logging.INFO)
for h in [logging.FileHandler(V2 / "logs" / "posthoc_mechanism.log"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s")); log.addHandler(h)


NONFINITE: dict = {}


def ci(pt, reps, name=None):
    """[estimate, 2.5%, 97.5%]. Non-finite replicates (empty group or log(0), only with small groups) are excluded
    and counted in NONFINITE[name]; with more than 2.5% non-finite replicates the CI is [nan, nan]."""
    reps = np.asarray(reps, float)
    ok = np.isfinite(reps)
    if name is not None and (~ok).any():
        NONFINITE[name] = int((~ok).sum())
    if (~ok).mean() > 0.025:
        return [float(pt), float("nan"), float("nan")]
    lo, hi = np.percentile(reps[ok], [2.5, 97.5])
    return [float(pt), float(lo), float(hi)]


def groups_of(pred, unk, stereo):
    u = pred == unk
    return {"S": ~u & (pred == stereo), "A": ~u & (pred != stereo), "U": u}


def raw_from(S, A):
    return 2 * S / (S + A) - 1


def analyse(m, split, sel):
    cb = sel["fairsteer"]["bbq"]
    L = int(cb["layer"])
    B, G, Un = load(m, split, "baseline"), load(m, split, tag("fairsteer", cb)), load(m, split, tag("fairsteer", cb, "_t-1"))
    for d in (G, Un):
        assert (d.uid.values == B.uid.values).all()
    bb = BBQ(B)
    amb = bb.amb
    cl, C = bb.cl, bb.C
    W = boot_weights(C, N_BOOT, seed=21)
    csum = lambda x: np.bincount(cl, weights=np.asarray(x, float), minlength=C)  # noqa: E731

    def rate(num, den):
        """Ratio of counts with a cluster-bootstrap CI."""
        n, d = csum(num), csum(den)
        return {"num": int(n.sum()), "den": int(d.sum()), "rate": ci(n.sum() / d.sum(), (W @ n) / (W @ d))}

    flag = G.flag.values.astype(bool)
    code = G.prob_biased.values.astype(np.float64)
    text = probe_scores(m, L, "probes_text", B.uid.values)
    tflag = text > 0.5
    gb = groups_of(B.pred.values, bb.unk, bb.stereo)
    gb = {g: v & amb for g, v in gb.items()}
    gg = {g: v & amb for g, v in groups_of(G.pred.values, bb.unk, bb.stereo).items()}
    gu = {g: v & amb for g, v in groups_of(Un.pred.values, bb.unk, bb.stereo).items()}
    # on unflagged items the gated run equals the baseline (check 1)
    assert (G.pred.values[~flag] == B.pred.values[~flag]).all()
    out = {"n_amb": int(amb.sum()), "counts": {g: int(gb[g].sum()) for g in GROUPS}}

    # ---------------------------------------------------------------- 1. selection
    out["selection"] = {spec: {g: rate(gb[g] & f, gb[g]) for g in GROUPS} for spec, f in [("code", flag), ("text", tflag)]}

    # ---------------------------------------------------------------- 1. conversion (ungated run)
    conv_ev = {"S": gu["U"], "A": gu["U"], "U": gu["S"] | gu["A"]}  # S,A -> unknown; U -> another answer
    out["conversion"] = {g: {"flagged": rate(gb[g] & flag & conv_ev[g], gb[g] & flag),
                             "unflagged": rate(gb[g] & ~flag & conv_ev[g], gb[g] & ~flag),
                             "all": rate(gb[g] & conv_ev[g], gb[g])} for g in GROUPS}

    # ---------------------------------------------------------------- 1. flows and decomposition of Δ raw_s_amb (code gate)
    F = {f"{a}{b}": gb[a] & gg[b] & flag for a in GROUPS for b in GROUPS if a != b}
    out["flows_flagged"] = {k: int(v.sum()) for k, v in F.items()}
    out["flows_flagged_stay"] = {g: int((gb[g] & gg[g] & flag).sum()) for g in GROUPS}
    cs = {k: csum(v) for k, v in F.items()}
    cs["S0"], cs["A0"] = csum(gb["S"]), csum(gb["A"])
    cs["fS"], cs["fA"] = csum(gb["S"] & flag), csum(gb["A"] & flag)

    def decomp(t):
        """t: dict of totals (scalars or arrays of replicates). Sequential decomposition: exits, swaps, entries."""
        S0, A0 = t["S0"], t["A0"]
        r0 = raw_from(S0, A0)
        S1, A1 = S0 - t["SU"], A0 - t["AU"]
        ra = raw_from(S1, A1)
        S2, A2 = S1 - t["SA"] + t["AS"], A1 - t["AS"] + t["SA"]
        rb = raw_from(S2, A2)
        S3, A3 = S2 + t["US"], A2 + t["UA"]
        rc = raw_from(S3, A3)
        rS, rA = t["SU"] / S0, t["AU"] / A0
        fS, fA = t["fS"] / S0, t["fA"] / A0
        cS, cA = t["SU"] / t["fS"], t["AU"] / t["fA"]
        return {"raw_base": r0, "d_raw_total": rc - r0, "d_raw_exits": ra - r0, "d_raw_swaps": rb - ra,
                "d_raw_entries": rc - rb, "r_S": rS, "r_A": rA, "f_S": fS, "f_A": fA, "c_S": cS, "c_A": cA,
                "log_r_ratio": np.log(rS / rA), "log_f_ratio": np.log(fS / fA), "log_c_ratio": np.log(cS / cA)}

    pt = decomp({k: v.sum() for k, v in cs.items()})
    rep = decomp({k: W @ v for k, v in cs.items()})
    NONFINITE.clear()
    out["decomposition"] = {k: ci(pt[k], rep[k], k) for k in pt}
    out["decomposition_nonfinite_reps"] = dict(NONFINITE)
    # check: total Δ = Δ raw_s_amb of the gated run
    Ib, Ig = bb.ind(B.pred.values), bb.ind(G.pred.values)
    d_true = BBQ.metrics(bb.csum(Ig).sum(0))["raw_s_amb"] - BBQ.metrics(bb.csum(Ib).sum(0))["raw_s_amb"]
    assert abs(d_true - pt["d_raw_total"]) < 1e-12, (d_true, pt["d_raw_total"])

    # ---------------------------------------------------------------- 2. does the probe tell the direction apart?
    nn = gb["S"] | gb["A"]
    y, s = gb["S"][nn].astype(int), code[nn]
    iw = W[:, cl[nn]]
    out["auroc_S_vs_A"] = {"code": ci(roc_auc_score(y, s), [roc_auc_score(y, s, sample_weight=w) for w in iw]),
                           "text": ci(roc_auc_score(y, text[nn]),
                                      [roc_auc_score(y, text[nn], sample_weight=w) for w in iw]),
                           "n_S": int(y.sum()), "n_A": int((1 - y).sum())}
    out["mean_score"] = {}
    for spec, sc in [("code", code), ("text", text)]:
        out["mean_score"][spec] = {}
        for g in GROUPS:
            num, den = csum(sc * gb[g]), csum(gb[g])
            out["mean_score"][spec][g] = ci(num.sum() / den.sum(), (W @ num) / (W @ den))

    # ---------------------------------------------------------------- 3. oracles at k_amb
    Iu = bb.ind(Un.pred.values)
    D = Iu - Ib
    Sb = bb.csum(Ib)
    n = len(B)
    base_pt, base_rep = BBQ.metrics(Sb.sum(0)), BBQ.metrics(W @ Sb)

    def S_of(g):
        return Sb + bb.csum(D * np.asarray(g, bool)[:, None])

    k_amb = int((flag & amb).sum())
    k = int(flag.sum())
    out["k"], out["k_amb"] = k, k_amb
    Scode = S_of(flag)
    code_pt, code_rep = BBQ.metrics(Scode.sum(0)), BBQ.metrics(W @ Scode)
    gates = {"code": {kk: ci(code_pt[kk] - base_pt[kk], code_rep[kk] - base_rep[kk]) for kk in AMB_METRICS}
             | {"n_amb_interventions": k_amb}}
    rng = np.random.default_rng(31)
    pools = {"oracle_nonunk": np.where(nn)[0], "oracle_ctx_kamb": np.where(amb)[0]}
    for name, pool in pools.items():
        size = min(k_amb, len(pool))
        sat = size == len(pool)
        draws = {kk: [] for kk in AMB_METRICS}
        diff_rep = {kk: np.empty(N_BOOT) for kk in AMB_METRICS}
        oracle_rep = {kk: np.empty(N_BOOT) for kk in AMB_METRICS}
        for _ in range(1 if sat else N_DRAWS):
            g = np.zeros(n, bool); g[rng.choice(pool, size=size, replace=False)] = True
            p = BBQ.metrics(S_of(g).sum(0))
            for kk in AMB_METRICS:
                draws[kk].append(p[kk] - base_pt[kk])
        # bootstrap: resampled clusters and, if not saturated, a new gate per replicate
        for i in range(N_BOOT):
            g = np.zeros(n, bool); g[pool if sat else rng.choice(pool, size=size, replace=False)] = True
            r = BBQ.metrics(W[i] @ S_of(g))
            for kk in AMB_METRICS:
                oracle_rep[kk][i] = r[kk] - base_rep[kk][i]
                diff_rep[kk][i] = (code_rep[kk][i] - base_rep[kk][i]) - oracle_rep[kk][i]
        o = {"n_amb_interventions": size, "pool": int(len(pool)), "saturated": bool(sat)}
        for kk in AMB_METRICS:
            mu = float(np.mean(draws[kk]))
            o[kk] = ci(mu, oracle_rep[kk])
            o[f"{kk}_draws"] = [mu, float(np.percentile(draws[kk], 2.5)), float(np.percentile(draws[kk], 97.5))]
            o[f"code_minus_{kk}"] = ci(gates["code"][kk][0] - mu, diff_rep[kk])
        gates[name] = o
    # deterministic gate: all stereotyped ambiguous items
    Sst = S_of(gb["S"])
    p, r = BBQ.metrics(Sst.sum(0)), BBQ.metrics(W @ Sst)
    gates["all_stereo"] = {kk: ci(p[kk] - base_pt[kk], r[kk] - base_rep[kk]) for kk in AMB_METRICS} \
        | {"n_amb_interventions": int(gb["S"].sum())}
    for kk in AMB_METRICS:
        gates["all_stereo"][f"code_minus_{kk}"] = ci((code_pt[kk] - base_pt[kk]) - (p[kk] - base_pt[kk]),
                                                     (code_rep[kk] - base_rep[kk]) - (r[kk] - base_rep[kk]))
    # effect per intervention on ambiguous items (s_amb points per 1,000 interventions)
    for name, gd in gates.items():
        gd["s_amb_per_1000"] = 1000 * gd["s_amb"][0] / gd["n_amb_interventions"]
        gd["raw_s_amb_per_1000"] = 1000 * gd["raw_s_amb"][0] / gd["n_amb_interventions"]
    out["oracles"] = gates

    # ---------------------------------------------------------------- 4. wasted interventions
    conf = confidence(B)
    g_conf = np.zeros(n, bool); g_conf[np.argsort(conf, kind="stable")[:k]] = True
    wU = gb["U"]
    waste = {}
    for name, g in [("code", flag), ("conf", g_conf)]:
        waste[name] = {"share_of_all": float((g & wU).sum() / g.sum()),
                       "share_of_amb": float((g & wU).sum() / (g & amb).sum()), "n": int(g.sum()), "n_amb": int((g & amb).sum())}
    # random gates: exact expected values
    waste["rand"] = {"share_of_all": float(wU.sum() / n), "share_of_amb": float(wU.sum() / amb.sum())}
    waste["oracle_ctx"] = {"share_of_all": float(wU.sum() / amb.sum()), "share_of_amb": float(wU.sum() / amb.sum())}
    waste["oracle_nonunk"] = {"share_of_all": 0.0, "share_of_amb": 0.0}
    out["wasted"] = waste
    return out


def main():
    models = sys.argv[1:] or ["llama", "mistral_instruct", "qwen"]
    for m in models:
        sel = json.load(open(RES / m / "report" / "selection.json"))
        res = {}
        for split in ("test", "dev"):
            r = analyse(m, split, sel)
            res[split] = r
            d = r["decomposition"]
            log.info(f"{m} {split}: n {r['counts']} k_amb {r['k_amb']} | f_code "
                     + " ".join(f"{g}={r['selection']['code'][g]['rate'][0]:.3f}" for g in GROUPS)
                     + " | c_flag " + " ".join(f"{g}={r['conversion'][g]['flagged']['rate'][0]:.3f}" for g in GROUPS)
                     + f" | Δraw {d['d_raw_total'][0]:+.3f} (exits {d['d_raw_exits'][0]:+.3f}, swaps {d['d_raw_swaps'][0]:+.3f},"
                     f" entries {d['d_raw_entries'][0]:+.3f}) | AUROC S/A {r['auroc_S_vs_A']['code'][0]:.3f}"
                     + " | s_amb code " + f"{r['oracles']['code']['s_amb'][0]:+.3f} nonunk {r['oracles']['oracle_nonunk']['s_amb'][0]:+.3f}"
                     f" (sat {r['oracles']['oracle_nonunk']['saturated']}) ctx@kamb {r['oracles']['oracle_ctx_kamb']['s_amb'][0]:+.3f}"
                     f" allS {r['oracles']['all_stereo']['s_amb'][0]:+.3f}")
        json.dump(res, open(RES / m / "report" / "posthoc_mechanism.json", "w"), indent=1)
        log.info(f"written {m}")


if __name__ == "__main__":
    main()
