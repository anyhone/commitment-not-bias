"""GPU regression tests. Slow: `pytest tests/test_gpu.py -x -q`.
The model is chosen with the environment variable STEER_TEST_MODEL (default: llama), e.g.
`STEER_TEST_MODEL=qwen pytest tests/test_gpu.py -q`."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from steering import data as D  # noqa: E402
from steering.experiment import make_steerer, run_bbq, run_ss  # noqa: E402
from steering.metrics import ss_metrics  # noqa: E402
from steering.model import bbq_user_text, load_config, load_model  # noqa: E402
from steering.scoring import ll_score, mc_score  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
LAYER = 14
MODEL = os.environ.get("STEER_TEST_MODEL", "llama")


@pytest.fixture(scope="session")
def mt():
    cfg = load_config(MODEL)
    model, tok = load_model(cfg)
    return model, tok, cfg


def _d(model):
    return model.config.get_text_config().hidden_size


@pytest.fixture(scope="session")
def small():
    b, s = D.with_splits(D.load_bbq(), D.load_stereoset())
    b = b[(b.split == "dev") & b.target_idx.notna()].sample(24, random_state=0).reset_index(drop=True)
    s = s[s.split == "dev"].sample(24, random_state=0).reset_index(drop=True)
    return b, s


def _rand_vec(d, norm=8.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return (v / v.norm() * norm).to("cuda", torch.float16)


def test_batch_invariance_mc(mt, small):
    model, tok, cfg = mt
    texts = [bbq_user_text(r) for _, r in small[0].iterrows()]
    a = mc_score(model, tok, cfg, texts, 3, batch_size=cfg["batch_size_mc"])
    b = mc_score(model, tok, cfg, texts, 3, batch_size=1)
    # numerical tolerance: on Qwen (DeltaNet kernels + 4-bit) the batch shape changes the rounding;
    # the absence of leakage from padding is checked separately in test_padding_no_leakage
    assert np.median(np.abs(a["logprob"] - b["logprob"])) < 0.01
    assert (a["pred"] != b["pred"]).sum() <= 1


def test_batch_invariance_ll(mt, small):
    model, tok, cfg = mt
    pairs = list(zip(small[1].ctx_stereo, small[1].comp_stereo))
    a = ll_score(model, tok, pairs, batch_size=cfg["batch_size_ll"])["ll"]
    b = ll_score(model, tok, pairs, batch_size=1)["ll"]
    assert np.median(np.abs(a - b)) < 0.02 and np.abs(a - b).max() < 0.1


def test_padding_no_leakage(mt, small):
    """The content of the (right) padding tokens must not affect the real tokens."""
    model, tok, cfg = mt
    from steering.scoring import encode_ll
    enc, _ = encode_ll(tok, list(zip(small[1].ctx_stereo, small[1].comp_stereo))[:8])
    g = torch.Generator().manual_seed(0)
    for ids, _ in enc:
        n = len(ids)
        x = torch.tensor(ids).cuda()
        pa = torch.cat([x, torch.full((32,), tok.pad_token_id).cuda()])
        pb = torch.cat([x, torch.randint(1000, 30000, (32,), generator=g).cuda()])
        with torch.no_grad():
            la = model(input_ids=pa[None]).logits[0, :n].float()
            lb = model(input_ids=pb[None]).logits[0, :n].float()
        assert torch.allclose(la, lb, atol=1e-3), "tokens after the sequence affect the earlier ones"


def test_caa_zero_coef_is_baseline(mt, small):
    model, tok, cfg = mt
    base, _ = run_ss(model, tok, cfg, small[1])
    st = make_steerer(model, "caa", LAYER, 0.0)
    try:
        z, _ = run_ss(model, tok, cfg, small[1], st, {c: _rand_vec(_d(model)) for c in D.SS_CATS})
    finally:
        st.remove()
    assert np.array_equal(base.ll_stereo.values, z.ll_stereo.values)


def test_steering_changes_stereoset(mt, small):
    """Regression test: steering must change the StereoSet log-likelihoods."""
    model, tok, cfg = mt
    base, _ = run_ss(model, tok, cfg, small[1])
    st = make_steerer(model, "caa", LAYER, 1.0)
    try:
        r, _ = run_ss(model, tok, cfg, small[1], st, {c: _rand_vec(_d(model)) for c in D.SS_CATS})
    finally:
        st.remove()
    assert (np.abs(r.ll_stereo.values - base.ll_stereo.values) > 1e-3).all()


_D = 4096  # hidden size shared by the three models (checked in test_hidden_size)


def test_hidden_size(mt):
    assert _d(mt[0]) == _D


def _probe_half():
    return {"mean": torch.zeros(_D, device="cuda"), "scale": torch.ones(_D, device="cuda"),
            "w": torch.zeros(_D, device="cuda"), "b": torch.tensor(0.0, device="cuda")}  # P = 0.5


def test_fairsteer_threshold_extremes(mt, small):
    """Threshold 0: steering on every item, identical to CAA restricted to the last token; threshold ~1: baseline."""
    model, tok, cfg = mt
    vec = {c: _rand_vec(_d(model)) for c in D.BBQ_CATS}
    base, _ = run_bbq(model, tok, cfg, small[0])
    out = {}
    for name, method, thr in [("always", "fairsteer", 0.0), ("never", "fairsteer", 0.99), ("caa_last", "caa", 0.5)]:
        st = make_steerer(model, method, LAYER, 1.0, _probe_half(), thr)
        if name == "caa_last":
            st.last_only = True
        try:
            out[name], _ = run_bbq(model, tok, cfg, small[0], st, vec)
        finally:
            st.remove()
    assert np.allclose(out["always"].lp0, out["caa_last"].lp0)
    assert not np.allclose(out["always"].lp0, base.lp0)
    assert np.allclose(out["never"].lp0, base.lp0)
    assert out["always"].flag.mean() == 1.0 and out["never"].flag.mean() == 0.0


def test_fairsteer_changes_stereoset(mt, small):
    """Regression test on the FairSteer path: steering at the last context token must
    change the completion log-likelihood."""
    model, tok, cfg = mt
    base, _ = run_ss(model, tok, cfg, small[1])
    st = make_steerer(model, "fairsteer", LAYER, 1.0, _probe_half(), 0.0)
    try:
        r, _ = run_ss(model, tok, cfg, small[1], st, {c: _rand_vec(_d(model)) for c in D.SS_CATS})
    finally:
        st.remove()
    assert (np.abs(r.ll_stereo.values - base.ll_stereo.values) > 1e-3).mean() > 0.9


def test_sign_of_dsv(mt):
    """With the extracted DSVs (pointing towards the bias) and coef < 0 the bias must decrease."""
    vp = V2 / "results" / MODEL / "vectors.pt"
    if not vp.exists():
        pytest.skip("vectors not extracted yet")
    model, tok, cfg = mt
    V = torch.load(vp, weights_only=False)
    b, s = D.with_splits(D.load_bbq(), D.load_stereoset())
    s = s[s.split == "dev"].sample(200, random_state=0).reset_index(drop=True)
    base, _ = run_ss(model, tok, cfg, s)
    st = make_steerer(model, "caa", LAYER, -4.0)
    try:
        g = V["ss"][LAYER]["__pooled__"]
        g = (g / g.norm() * V["meta"]["resid_norm"][LAYER]).to("cuda", torch.float16)
        st.coef = -0.3
        r, _ = run_ss(model, tok, cfg, s, st, {c: g for c in D.SS_CATS})
    finally:
        st.remove()
    assert ss_metrics(r)["gap"] < ss_metrics(base)["gap"]
