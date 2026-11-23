"""Diagnosis (analysis only, NOT used for selection nor evaluated on the test set): why steering
the last prompt token (FairSteer, Eq. 4) barely moves the answers on some models (Qwen3.5).

Fixed sample of ambiguous dev BBQ items (same seed for all models). For the requested layers:
 A. residual-stream norms per position: last prompt token (where FairSteer intervenes),
    answer letter token (reference of the relative coefficient, meta.resid_norm of the vectors),
    positions after the instruction (where CAA intervenes), maximum and first position;
 B. effect of steering with the global BBQ DSV, WITHOUT the probe (all items), per set of positions:
      last      = last token only, scale resid_norm (as FairSteer with the flag always on)
      last_own  = last token only, scale = the item's own last-token norm
      caa       = all positions after the instruction, scale resid_norm (as CAA)
    metrics vs baseline: % changed predictions, % unknown (= accuracy on ambiguous items),
    mean Δ of the log-prob of the unknown option;
 C. propagation (last mode): ‖Δh‖ at the last token in the following layers / injected ‖Δ‖;
 D. cosine between the global DSV and the BAD probe direction (w / scale, activation space).

usage: python scripts/diag_last_token.py --model qwen [--n 400] [--layers 12 16 20 24]
output: results/<model>/diag/last_token.json + log logs/<model>_diag.log
"""
import argparse
import json

import numpy as np
import torch

from _common import D, get_logger, load_vectors, mdir
from steering.hooks import Capturer, Steerer
from steering.model import bbq_user_text, get_layers, hidden_of, load_config, load_model
from steering.scoring import encode_mc, mc_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--n-norm", type=int, default=50, help="items for the per-position norms (batch 1)")
ap.add_argument("--n-prop", type=int, default=100, help="items for the propagation")
ap.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20, 24])
ap.add_argument("--coefs", type=float, nargs="+", default=[-0.3, -0.8])
a = ap.parse_args()

cfg = load_config(a.model)
log = get_logger(a.model, "diag")
model, tok = load_model(cfg)
OUT = mdir(a.model) / "diag"; OUT.mkdir(exist_ok=True)
L = a.layers
nL = len(get_layers(model))
V = load_vectors(a.model)
rn = {l: V["meta"]["resid_norm"][l] for l in L}
unit = {l: (V["bbq"][l]["__pooled__"] / V["bbq"][l]["__pooled__"].norm()).cuda().half() for l in L}

b, _ = D.with_splits(D.load_bbq(), D.load_stereoset())
b = b[(b.split == "dev") & b.target_idx.notna() & (b.context_condition == "ambig")]
b = b.sample(n=a.n, random_state=D.SEED).reset_index(drop=True)
texts = [bbq_user_text(r) for _, r in b.iterrows()]
enc = encode_mc(tok, cfg, texts)
unk = b.unknown_idx.astype(int).values
R = {"model": a.model, "n": a.n, "layers": L, "coefs": a.coefs, "n_layers": nL,
     "prompt_tail": tok.convert_ids_to_tokens(enc[0][0][-8:]), "resid_norm_letter": rn}
log.info(f"diagnosis {a.model}: {a.n} ambiguous dev items, layers {L}, prompt tail {R['prompt_tail']}")


def save():
    (OUT / "last_token.json").write_text(json.dumps(R, indent=1, default=float))


# ---------------------------------------------------------------- A. norms per position
cur = {}
hs = [get_layers(model)[l].register_forward_hook(
    (lambda l: lambda m, i, o: cur.__setitem__(l, hidden_of(o)[0].float().norm(dim=-1).cpu()))(l)) for l in L]
A = {l: {"last": [], "after_instr": [], "user": [], "first": [], "max": [], "argmax_from_end": []} for l in L}
with torch.no_grad():
    for i in range(min(a.n_norm, a.n)):
        ids, start = enc[i]
        model(input_ids=torch.tensor([ids], device=model.device))
        for l in L:
            n = cur[l]
            A[l]["last"].append(float(n[-1])); A[l]["first"].append(float(n[0])); A[l]["max"].append(float(n.max()))
            A[l]["after_instr"].append(float(n[start:].mean())); A[l]["user"].append(float(n[1:start].median()))
            A[l]["argmax_from_end"].append(int(len(n) - 1 - int(n.argmax())))
for h in hs:
    h.remove()
R["norms"] = {l: {k: (float(np.mean(v)) if k != "argmax_from_end" else int(np.median(v))) for k, v in A[l].items()}
              for l in L}
for l in L:
    x = R["norms"][l]
    log.info(f"A L{l}: |h| last token {x['last']:.1f} | letter token (ref.) {rn[l]:.1f} | ratio {x['last'] / rn[l]:.2f} "
             f"| after instruction {x['after_instr']:.1f} | user (median) {x['user']:.1f} | first {x['first']:.1f} "
             f"| max {x['max']:.1f} (median dist. from the end {x['argmax_from_end']})")
save()

# ---------------------------------------------------------------- B. effect per set of positions
cap = Capturer(model, L)
base = mc_score(model, tok, cfg, texts, 3, capturer=cap, batch_size=cfg["batch_size_mc"], desc="diag:base")
cap.remove()
last_norm = {l: base["acts"][l].float().norm(dim=1) for l in L}
lp0, p0 = base["logprob"], base["pred"]
R["baseline"] = {"unknown_rate": float((p0 == unk).mean() * 100)}
log.info(f"B baseline: unknown {R['baseline']['unknown_rate']:.1f}%")
R["effect"] = []
for l in L:
    for c in a.coefs:
        for mode in ["last", "last_own", "caa"]:
            st = Steerer(model, l)
            st.mode, st.coef, st.last_only = "caa", c, mode != "caa"
            if mode == "last_own":
                vecs = [unit[l] * float(last_norm[l][i]) for i in range(a.n)]
            else:
                vecs = unit[l] * rn[l]
            try:
                r = mc_score(model, tok, cfg, texts, 3, st, vecs, batch_size=cfg["batch_size_mc"], desc=f"diag:{mode}:L{l}{c:g}")
            finally:
                st.remove()
            lp, p = r["logprob"], r["pred"]
            rows = np.arange(a.n)
            e = {"layer": l, "coef": c, "mode": mode,
                 "changed": float((p != p0).mean() * 100),
                 "unknown_rate": float((p == unk).mean() * 100),
                 "d_lp_unknown": float((lp[rows, unk] - lp0[rows, unk]).mean()),
                 "max_abs_d_lp_median": float(np.median(np.abs(lp - lp0).max(1)))}
            R["effect"].append(e)
            log.info(f"B L{l} s{c:g} {mode:8s}: changed {e['changed']:5.1f}% | unknown {e['unknown_rate']:5.1f}% "
                     f"| Δlogp(unknown) {e['d_lp_unknown']:+.3f} | median max|Δlogp| {e['max_abs_d_lp_median']:.3f}")
            save()

# ---------------------------------------------------------------- C. propagation (last mode)
m = min(a.n_prop, a.n)
R["propagation"] = {}
for l in L:
    later = list(range(l, nL))
    c = min(a.coefs)
    cap0 = Capturer(model, later)
    h0 = mc_score(model, tok, cfg, texts[:m], 3, capturer=cap0, batch_size=cfg["batch_size_mc"], desc=f"diag:prop0:L{l}")["acts"]
    cap0.remove()
    st = Steerer(model, l)
    st.mode, st.coef, st.last_only = "caa", c, True
    cap1 = Capturer(model, later)  # registered after the Steerer: at layer l it sees the already modified output
    try:
        h1 = mc_score(model, tok, cfg, texts[:m], 3, st, unit[l] * rn[l], capturer=cap1,
                      batch_size=cfg["batch_size_mc"], desc=f"diag:prop1:L{l}")["acts"]
    finally:
        cap1.remove(); st.remove()
    inj = abs(c) * rn[l]
    prop = {}
    for k in later:
        d = (h1[k].float() - h0[k].float())
        prop[k] = {"rel_injected": float(d.norm(dim=1).mean() / inj),
                   "rel_h": float((d.norm(dim=1) / h0[k].float().norm(dim=1)).mean()),
                   "cos_dsv": float(torch.nn.functional.cosine_similarity(d, unit[l].float().cpu()[None], dim=1).mean())}
    R["propagation"][l] = prop
    ks = [k for k in later if k in (l, l + 2, l + 4, nL - 5, nL - 2, nL - 1)]
    log.info(f"C L{l} s{c:g}: ‖Δh‖/‖injected Δ‖ " + " ".join(f"L{k}={prop[k]['rel_injected']:.2f}" for k in ks) +
             " | ‖Δh‖/‖h‖ " + " ".join(f"L{k}={prop[k]['rel_h']:.3f}" for k in ks) +
             " | cos(Δh, DSV) " + " ".join(f"L{k}={prop[k]['cos_dsv']:.2f}" for k in ks))
    save()

# ---------------------------------------------------------------- D. DSV vs probe
R["cos_dsv_probe"] = {}
for l in L:
    f = mdir(a.model) / "probes" / f"probe_L{l}.pt"
    if f.exists():
        p = torch.load(f, weights_only=False)["torch"]
        w = (p["w"].float().flatten() / p["scale"].float().flatten()).cpu()
        R["cos_dsv_probe"][l] = float(torch.nn.functional.cosine_similarity(w, unit[l].float().cpu(), dim=0))
log.info(f"D cosine global DSV / probe direction: { {l: round(v, 3) for l, v in R['cos_dsv_probe'].items()} }")
save()
log.info(f"diagnosis done: {OUT / 'last_token.json'}")
