"""Tests on data, labels and splits (CPU)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steering import data as D  # noqa: E402


def test_bbq_target_coverage():
    b = D.load_bbq()
    assert len(b) == 31372
    assert b.target_idx.isna().sum() <= 16  # only items whose two answers have the same gender


def test_bbq_stereo_semantics():
    b = D.load_bbq().dropna(subset=["target_idx"])
    assert (b.stereo_idx != b.unknown_idx).all()
    assert (b.target_idx != b.unknown_idx).all()
    neg = b.question_polarity == "neg"
    assert (b.loc[neg, "stereo_idx"] == b.loc[neg, "target_idx"]).all()
    assert (b.loc[~neg, "stereo_idx"] != b.loc[~neg, "target_idx"]).all()


def test_bbq_ambig_label_is_unknown():
    b = D.load_bbq()
    a = b[b.context_condition == "ambig"]
    assert (a.label == a.unknown_idx).all()


def test_splits_disjoint():
    b, s = D.with_splits(D.load_bbq(), D.load_stereoset())
    for col, df in [("template", b), ("target", s)]:
        g = df.groupby(col if col == "template" else ["bias_type", "target"]).split.nunique()
        assert (g == 1).all(), f"{col} keys in more than one split"
    assert set(b.split) == {"train", "dev", "test"} and set(s.split) == {"train", "dev", "test"}


def test_stereoset_labels():
    s = D.load_stereoset()
    assert len(s) == 4229
    for o in ["stereo", "anti", "unrelated"]:
        assert s[f"comp_{o}"].str.len().gt(0).all()
    # the three completions share the same context
    assert (s.ctx_stereo == s.ctx_anti).all() and (s.ctx_stereo == s.ctx_unrelated).all()


def test_mmlu_neutral_not_in_test():
    aux = D.mmlu_neutral_pool()
    test = D.load_mmlu("test")
    assert not set(aux.question) & set(test.question)


def test_mmlu_neutral_three_options_no_d():
    """As in FairSteer (App. C.1): MMLU neutral examples without D answers and without option D."""
    aux = D.mmlu_neutral_pool()
    assert (aux.answer != 3).all()
    assert aux.choices.apply(len).eq(3).all()
