"""Post-hoc analysis of the monitor, added after the test evaluation (protocol written before these runs).

For each model and split (dev, test), at the FairSteer configuration selected for BBQ:
  - Check 1: gated run == where(flag, ungated run, baseline) (BBQ, StereoSet, MMLU);
  - Check 2: flag recomputed from the baseline activations == flag saved in the gated run;
  - simulated gates at rate k (code-specification monitor, text specification, random, confidence, context oracle);
  - primary hypotheses: code monitor − random / − confidence, Δ acc_dis and Δ s_amb, Holm over 6 tests per metric;
  - 2x2 decomposition position x monitor (also at the StereoSet configuration, if available);
  - threshold curve (descriptive), probe vs confidence, "unknown" rates of the existing conditions.
Parts whose runs are missing are skipped (and logged).

Usage: uv run python scripts/posthoc_gates.py [model ...]
Output: results/<model>/report/posthoc.json, posthoc_threshold_curve_{dev,test}.csv
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from steering.metrics import bbq_metrics, paired_bootstrap, ss_metrics  # noqa: E402

RES = V2 / "results"
N_CI, N_PRIMARY, N_DRAWS = 1000, 10000, 1000
METRICS = ["s_amb", "raw_s_amb", "acc_amb", "acc_dis", "unk_amb", "unk_dis"]
TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)

(V2 / "logs").mkdir(exist_ok=True)
log = logging.getLogger("posthoc")
log.setLevel(logging.INFO)
for h in [logging.FileHandler(V2 / "logs" / "posthoc_gates.log"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s")); log.addHandler(h)


def tag(meth, cfg, suffix=""):
    return f"{meth}_L{int(cfg['layer'])}_{cfg.get('coef_mode', 'rel')}{cfg['coef']:g}{suffix}"


def load(m, split, cond, kind="bbq"):
    p = RES / m / "eval" / split / cond / f"{kind}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ------------------------------------------------------------------ vectorised BBQ metrics per cluster
class BBQ:
    """Per-item indicators and per-cluster (template) sums, for fast metrics and bootstrap.
    Same definition as steering.metrics.bbq_metrics: 'biased' = pred == stereo_idx among non-unknown answers."""

    def __init__(self, ref: pd.DataFrame):
        self.uid = ref.uid.values
        self.amb = (ref.context_condition == "ambig").values
        self.label, self.unk, self.stereo = ref.label.values, ref.unknown_idx.values, ref.stereo_idx.astype(int).values
        self.tpl_names, self.cl = np.unique(ref.template.values, return_inverse=True)
        self.C = len(self.tpl_names)

    def ind(self, pred):
        """Items x indicators matrix: [amb, amb&corr, amb&nonunk, amb&nonunk&stereo, amb&unk, dis, dis&corr, dis&unk]."""
        a, corr, unk = self.amb, pred == self.label, pred == self.unk
        nn = ~unk
        return np.stack([a, a & corr, a & nn, a & nn & (pred == self.stereo), a & unk,
                         ~a, ~a & corr, ~a & unk], 1).astype(np.int64)

    def csum(self, I):
        return np.stack([np.bincount(self.cl, weights=I[:, j], minlength=self.C) for j in range(I.shape[1])], 1)

    @staticmethod
    def metrics(S):
        """S: [..., 8] sums (possibly weighted). Returns a dict of arrays."""
        acc_amb = S[..., 1] / S[..., 0]
        raw = 2 * S[..., 3] / S[..., 2] - 1
        return {"s_amb": (1 - acc_amb) * raw, "raw_s_amb": raw, "acc_amb": 100 * acc_amb,
                "acc_dis": 100 * S[..., 6] / S[..., 5], "unk_amb": 100 * S[..., 4] / S[..., 0],
                "unk_dis": 100 * S[..., 7] / S[..., 5]}


def boot_weights(C, n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(n)]).astype(np.float64)


def summarize(point: dict, reps: dict) -> dict:
    out = {}
    for k in METRICS:
        lo, hi = np.nanpercentile(reps[k], [2.5, 97.5])
        out[k] = [float(point[k]), float(lo), float(hi)]
    return out


def pval(d):
    d = d[~np.isnan(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def probe_scores(m, layer, probe_set, uids):
    """P(biased) from the probe (torch format: mean, scale, w, b) on the saved baseline activations."""
    A = RES / m / "baseline" / "acts"
    all_uids = np.load(A / "bbq_uids.npy", allow_pickle=True)
    pos = pd.Series(np.arange(len(all_uids)), index=all_uids).loc[uids].values
    X = np.load(A / f"bbq_L{layer}.npy", mmap_mode="r")[pos].astype(np.float32)
    p = {k: v.float().numpy() for k, v in torch.load(RES / m / probe_set / f"probe_L{layer}.pt", weights_only=False)["torch"].items()}
    z = ((X - p["mean"]) / p["scale"]) @ p["w"] + p["b"]
    return 1 / (1 + np.exp(-z))


def confidence(df):
    lp = df[["lp0", "lp1", "lp2"]].values.astype(np.float64)
    pr = np.exp(lp - lp.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    return pr.max(1)


# ------------------------------------------------------------------ checks
def verify_sim(real, sim_on, base, flag, kind):
    """Agreement between the real (gated) run and where(flag, sim_on, base)."""
    flag = np.asarray(flag, bool)
    if kind == "bbq":
        cols = ["lp0", "lp1", "lp2"]
        sim_pred = np.where(flag, sim_on.pred.values, base.pred.values)
        agree = float((sim_pred == real.pred.values).mean())
    elif kind == "ss":
        cols = ["ll_stereo", "ll_anti", "ll_unrelated"]
        s = np.where(flag, (sim_on.ll_stereo > sim_on.ll_anti).values, (base.ll_stereo > base.ll_anti).values)
        agree = float((s == (real.ll_stereo > real.ll_anti).values).mean())
    else:
        cols = []
        sim_pred = np.where(flag, sim_on.pred.values, base.pred.values)
        agree = float((sim_pred == real.pred.values).mean())
    out = {"n": int(len(real)), "n_flag": int(flag.sum()), "agree": agree}
    if cols:
        sim = np.where(flag[:, None], sim_on[cols].values, base[cols].values)
        d = np.abs(sim - real[cols].values).max(1)
        out |= {"absdiff_median": float(np.median(d)), "absdiff_max": float(d.max()),
                "absdiff_median_flagged": float(np.median(d[flag])) if flag.any() else None}
    return out


# ------------------------------------------------------------------ analysis per split
def analyse_split(m, split, sel, R):
    cb = sel["fairsteer"]["bbq"]
    L = int(cb["layer"])
    B, G = load(m, split, "baseline"), load(m, split, tag("fairsteer", cb))
    U, C = load(m, split, tag("fairsteer", cb, "_t-1")), load(m, split, tag("caa", cb))
    P = load(m, split, tag("fairsteer", cb, "_probes_text"))
    out = {"layer": L, "available": {"ungated": U is not None, "caa_at_fs": C is not None, "text_run": P is not None}}
    for d in (G, U, C, P):
        if d is not None:
            assert (d.uid.values == B.uid.values).all()
    bb = BBQ(B)
    flag = G.flag.values.astype(bool)
    k = int(flag.sum())
    out["k"], out["rate"] = k, k / len(B)

    # probe scores: code specification (saved in the gated run) and text specification (offline)
    code_saved = G.prob_biased.values.astype(np.float64)
    code_off = probe_scores(m, L, "probes", B.uid.values)
    text_off = probe_scores(m, L, "probes_text", B.uid.values)
    v2 = {"n": len(B), "flag_agree": float(((code_off > 0.5) == flag).mean()),
          "n_flag_mismatch": int(((code_off > 0.5) != flag).sum()),
          "prob_absdiff_median": float(np.median(np.abs(code_off - code_saved))),
          "prob_absdiff_max": float(np.abs(code_off - code_saved).max())}
    if U is not None:
        v2["ungated_prob_equal_gated"] = float(np.abs(U.prob_biased.values - code_saved).max())
    if P is not None:
        v2["text_flag_agree"] = float(((text_off > 0.5) == P.flag.values.astype(bool)).mean())
        v2["text_prob_absdiff_max"] = float(np.abs(text_off - P.prob_biased.values).max())
    out["verify2"] = v2
    log.info(f"{m} {split} check 2: {v2}")

    # probe vs confidence
    conf = confidence(B)
    pred, lab, st = B.pred.values, B.label.values, B.stereo_idx.astype(int).values
    y = np.where((pred == st) & (pred != lab), 1, np.where(pred == lab, 0, -1))
    ky = y >= 0
    out["confidence"] = {}
    for spec, sc in [("code", code_saved), ("text", text_off)]:
        out["confidence"][spec] = {
            "spearman": float(spearmanr(sc, conf).correlation),
            "auroc_probe_biased": float(roc_auc_score(y[ky], sc[ky])),
            "auroc_conf_biased": float(roc_auc_score(y[ky], -conf[ky])),
            "auroc_probe_amb": float(roc_auc_score(bb.amb, sc)),
            "auroc_conf_amb": float(roc_auc_score(bb.amb, -conf)),
            "frac_biased": float(y[ky].mean())}
    log.info(f"{m} {split} confidence: {out['confidence']}")

    W = boot_weights(bb.C, N_CI, seed=1)
    Ib = bb.ind(B.pred.values)
    Sb = bb.csum(Ib)
    base_pt, base_rep = bb.metrics(Sb.sum(0)), bb.metrics(W @ Sb)

    def delta(S):
        pt, rep = bb.metrics(S.sum(0)), bb.metrics(W @ S)
        return summarize({k: pt[k] - base_pt[k] for k in METRICS}, {k: rep[k] - base_rep[k] for k in METRICS})

    def gate_stats(g):
        g = np.asarray(g, bool)
        return {"n": int(g.sum()), "rate": float(g.mean()), "rate_amb": float(g[bb.amb].mean()),
                "rate_dis": float(g[~bb.amb].mean())}

    out["base"] = {k: float(v) for k, v in base_pt.items()}

    # -------------------------------------------------------------- 2x2 position x monitor
    if U is not None:
        Iu = bb.ind(U.pred.values)
        Ig = bb.ind(G.pred.values)
        Sg, Su = bb.csum(Ig), bb.csum(Iu)
        out["verify1_bbq"] = verify_sim(G, U, B, flag, "bbq")
        log.info(f"{m} {split} check 1 BBQ: {out['verify1_bbq']}")
        cells = {"fs": Sg, "fsng": Su}
        if C is not None:
            Ic = bb.ind(C.pred.values)
            Icg = np.where(flag[:, None], Ic, Ib)
            cells |= {"caafs": bb.csum(Ic), "caafs_mon": bb.csum(Icg)}
        x = {c: delta(S) for c, S in cells.items()}
        if C is not None:
            def diff(S1, S0, S1b=None, S0b=None):
                pt1, pt0 = bb.metrics(S1.sum(0)), bb.metrics(S0.sum(0))
                r1, r0 = bb.metrics(W @ S1), bb.metrics(W @ S0)
                if S1b is None:
                    return summarize({k: pt1[k] - pt0[k] for k in METRICS}, {k: r1[k] - r0[k] for k in METRICS})
                q1, q0 = bb.metrics(S1b.sum(0)), bb.metrics(S0b.sum(0))
                s1, s0 = bb.metrics(W @ S1b), bb.metrics(W @ S0b)
                return summarize({k: (pt1[k] - pt0[k]) - (q1[k] - q0[k]) for k in METRICS},
                                 {k: (r1[k] - r0[k]) - (s1[k] - s0[k]) for k in METRICS})
            S = cells
            x["pos_effect_nomon"] = diff(S["caafs"], S["fsng"])
            x["pos_effect_mon"] = diff(S["caafs_mon"], S["fs"])
            x["mon_effect_last"] = diff(S["fs"], S["fsng"])
            x["mon_effect_all"] = diff(S["caafs_mon"], S["caafs"])
            x["interaction"] = diff(S["fs"], S["fsng"], S["caafs_mon"], S["caafs"])
        cref = load(m, split, tag("caa", sel["caa"]["bbq"]))
        if cref is not None:
            x["caa_ref"] = delta(bb.csum(bb.ind(cref.pred.values)))
        out["x2x2_bbq"] = x

        # ---------------------------------------------------------- simulated gates at rate k
        D = Iu - Ib  # change of the indicators if the item is steered
        rng = np.random.default_rng(7)
        amb_idx, n = np.where(bb.amb)[0], len(B)

        def S_of(g):
            return Sb + bb.csum(D * np.asarray(g, bool)[:, None])

        def topk(score):
            g = np.zeros(n, bool); g[np.argsort(-score, kind="stable")[:k]] = True; return g

        det = {"code": flag, "text": text_off > 0.5, "text_matched": topk(text_off), "conf": topk(-conf)}
        gates = {}
        for name, g in det.items():
            gates[name] = delta(S_of(g)) | gate_stats(g)
        # random and oracle: distribution over N_DRAWS draws (full data)
        for name, pool in [("rand", np.arange(n)), ("oracle", amb_idx)]:
            vals = {kk: [] for kk in METRICS}
            ra = []
            for _ in range(N_DRAWS):
                g = np.zeros(n, bool); g[rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
                pt = bb.metrics(S_of(g).sum(0))
                for kk in METRICS:
                    vals[kk].append(pt[kk] - base_pt[kk])
                ra.append(g[bb.amb].mean())
            gates[name] = {kk: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                           for kk, v in vals.items()} | {"n": k, "rate": k / n, "rate_amb": float(np.mean(ra)),
                                                          "rate_dis": float(k / n * 2 - np.mean(ra))}
        out["gates"] = gates

        # ---------------------------------------------------------- primary hypotheses (10,000 replicates)
        Wp = boot_weights(bb.C, N_PRIMARY, seed=11)
        Scode = S_of(flag)
        rc = bb.metrics(Wp @ Scode)
        Sconf = S_of(det["conf"])
        rf = bb.metrics(Wp @ Sconf)
        # random: a new gate per replicate
        rr = {kk: np.empty(N_PRIMARY) for kk in METRICS}
        rng2 = np.random.default_rng(13)
        for i in range(N_PRIMARY):
            g = np.zeros(n, bool); g[rng2.choice(n, size=k, replace=False)] = True
            mt = bb.metrics(Wp[i] @ S_of(g))
            for kk in METRICS:
                rr[kk][i] = mt[kk]
        pc = bb.metrics(Scode.sum(0))
        prim = {}
        for comp, other_rep, other_pt in [("code_minus_rand", rr, {kk: gates["rand"][kk][0] + base_pt[kk] for kk in METRICS}),
                                          ("code_minus_conf", rf, bb.metrics(Sconf.sum(0)))]:
            prim[comp] = {}
            for kk in METRICS:
                d = rc[kk] - other_rep[kk]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                prim[comp][kk] = {"diff": float(pc[kk] - other_pt[kk]), "ci": [float(lo), float(hi)], "p": pval(d)}
        out["primary"] = prim
        log.info(f"{m} {split} primary: " + json.dumps({c: {kk: prim[c][kk] for kk in ('acc_dis', 's_amb')} for c in prim}))

        # ---------------------------------------------------------- threshold curve
        rows = []
        for spec, sc in [("code", code_saved), ("text", text_off)]:
            for t in TAUS:
                g = sc > t
                pt = bb.metrics(S_of(g).sum(0))
                rows.append({"model": m, "split": split, "spec": spec, "tau": float(t), **gate_stats(g),
                             **{f"d_{kk}": float(pt[kk] - base_pt[kk]) for kk in METRICS},
                             **{kk: float(pt[kk]) for kk in ("unk_amb", "unk_dis")}})
        out["_curve"] = rows
    else:
        log.info(f"{m} {split}: ungated run not available yet, skipping 2x2, gates and curve")
    return out


def x2x2_other(m, split, sel, bench_cfg):
    """2x2 on StereoSet and MMLU (secondary) at the given configuration."""
    cfg = sel["fairsteer"][bench_cfg]
    src = bench_cfg
    B = {"ss": load(m, split, "baseline", "ss"), "mmlu": load(m, split, "baseline", "mmlu_none")}
    G = {"ss": load(m, split, tag("fairsteer", cfg), "ss"), "mmlu": load(m, split, tag("fairsteer", cfg), f"mmlu_{src}")}
    U = {"ss": load(m, split, tag("fairsteer", cfg, "_t-1"), "ss"), "mmlu": load(m, split, tag("fairsteer", cfg, "_t-1"), f"mmlu_{src}")}
    C = {"ss": load(m, split, tag("caa", cfg), "ss"), "mmlu": load(m, split, tag("caa", cfg), f"mmlu_{src}")}
    if U["ss"] is None or G["ss"] is None:
        return None
    # MMLU with the vector from the same source in all cells; on dev at the SS configuration the sweep has only
    # mmlu_bbq and the ungated run only mmlu_ss: in that case the MMLU part is skipped.
    has_mmlu = G["mmlu"] is not None and U["mmlu"] is not None
    if not has_mmlu:
        log.info(f"{m} {split} config {bench_cfg}: MMLU with the {src} vector not available in all cells, skipping MMLU")
    out = {"config": tag("fairsteer", cfg), "mmlu_source": src if has_mmlu else None,
           "verify1_ss": verify_sim(G["ss"], U["ss"], B["ss"], G["ss"].flag.values, "ss"),
           "verify1_mmlu": verify_sim(G["mmlu"], U["mmlu"], B["mmlu"], G["mmlu"].flag.values, "mmlu") if has_mmlu else None}
    # SS: cells as DataFrames (simulated cell with the flag of the gated run)
    ssc = {"fs": G["ss"], "fsng": U["ss"]}
    if C["ss"] is not None:
        f = G["ss"].flag.values.astype(bool)[:, None]
        cols = ["ll_stereo", "ll_anti", "ll_unrelated"]
        cm = C["ss"].copy(); cm[cols] = np.where(f, C["ss"][cols].values, B["ss"][cols].values)
        ssc |= {"caafs": C["ss"], "caafs_mon": cm}
    out["ss"] = {c: {kk: list(paired_bootstrap(ss_metrics, B["ss"], d, kk, n=N_CI, cluster="target").values())
                     for kk in ("SS", "LMS")} for c, d in ssc.items()}
    if not has_mmlu:
        out["mmlu"] = None
        return out
    acc = lambda d: float((d.pred == d.answer).mean() * 100)  # noqa: E731
    mm = {"fs": G["mmlu"], "fsng": U["mmlu"]}
    if C["mmlu"] is not None:
        cm = C["mmlu"].copy(); cm["pred"] = np.where(G["mmlu"].flag.values.astype(bool), C["mmlu"].pred, B["mmlu"].pred)
        mm |= {"caafs": C["mmlu"], "caafs_mon": cm}
    out["mmlu"] = {c: {"acc": acc(d), "d_acc": acc(d) - acc(B["mmlu"]),
                       "rate": float(np.nanmean(G["mmlu"].flag)) if c == "fs" else None} for c, d in mm.items()}
    return out


def unknown_rates(m):
    """'Unknown' rate per context for all test conditions."""
    out = {}
    for d in sorted((RES / m / "eval" / "test").iterdir()):
        df = load(m, "test", d.name)
        if df is None:
            continue
        df = df[df.target_idx.notna()]
        u = df.pred == df.unknown_idx
        out[d.name] = {"unk_amb": float(100 * u[df.context_condition == "ambig"].mean()),
                       "unk_dis": float(100 * u[df.context_condition == "disambig"].mean())}
    return out


def check_metrics(m):
    """The vectorised function matches bbq_metrics."""
    for cond in ["baseline", tag("fairsteer", json.load(open(RES / m / "report" / "selection.json"))["fairsteer"]["bbq"])]:
        df = load(m, "test", cond)
        bb = BBQ(df)
        v = bb.metrics(bb.csum(bb.ind(df.pred.values)).sum(0))
        r = bbq_metrics(df)
        for k in ["s_amb", "raw_s_amb", "acc_amb", "acc_dis"]:
            assert abs(v[k] - r[k]) < 1e-9, (m, cond, k, v[k], r[k])


def main():
    models = sys.argv[1:] or ["llama", "mistral_instruct", "qwen"]
    for m in models:
        check_metrics(m)
        sel = json.load(open(RES / m / "report" / "selection.json"))
        res = {"unknown_rates_test": unknown_rates(m)}
        curves = []
        for split in ("dev", "test"):
            r = analyse_split(m, split, sel, None)
            curves += r.pop("_curve", [])
            r["x2x2_other"] = {b: x2x2_other(m, split, sel, b) for b in ("bbq", "ss")}
            res[split] = r
        for split in ("dev", "test"):
            c = [x for x in curves if x["split"] == split]
            if c:
                pd.DataFrame(c).to_csv(RES / m / "report" / f"posthoc_threshold_curve_{split}.csv", index=False)
        json.dump(res, open(RES / m / "report" / "posthoc.json", "w"), indent=1)
        log.info(f"written {m}")
    # Holm on the primary hypotheses (test): 6 tests per metric
    prim = {}
    for m in models:
        r = json.load(open(RES / m / "report" / "posthoc.json"))["test"]
        if "primary" in r:
            prim[m] = r["primary"]
    if len(prim) == len(models) == 3:
        for kk in ("acc_dis", "s_amb"):
            keys = [(m, c) for m in models for c in ("code_minus_rand", "code_minus_conf")]
            adj = holm([prim[m][c][kk]["p"] for m, c in keys])
            for (m, c), a in zip(keys, adj):
                prim[m][c][kk]["p_holm"] = a
        for m in models:
            p = RES / m / "report" / "posthoc.json"
            r = json.load(open(p)); r["test"]["primary"] = prim[m]; json.dump(r, open(p, "w"), indent=1)
        log.info("Holm applied to the primary hypotheses")
    else:
        log.info("Holm not applied: primary hypotheses not available for all three models")


if __name__ == "__main__":
    main()
