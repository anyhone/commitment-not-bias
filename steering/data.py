"""Dataset loading and disjoint splits.

Conventions:
- BBQ: for each item we derive `target_idx` (answer naming the stereotyped group),
  `unknown_idx` and `stereo_idx` (stereotype-consistent answer: the target if the question is
  negative, the other group if it is non-negative). The split is by template (category:question_index).
- StereoSet: split by (bias_type, target).
- MMLU: test for the final evaluation, validation for dev, auxiliary_train for the probe's neutral examples.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[1]
DATA = V2 / "data"
BBQ_DIR = DATA / "bbq"  # jsonl files of the official BBQ repository (scripts/download_data.py)
SPLITS_FILE = DATA / "splits.json"

BBQ_CATS = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
            "Race_ethnicity", "Religion", "SES", "Sexual_orientation"]
SS_CATS = ["gender", "profession", "race", "religion"]
SPLIT_FRACS = {"train": 0.30, "dev": 0.20, "test": 0.50}
SEED = 1234

# synonyms between the `stereotyped_groups` labels and the `answer_info` tags
_SYN = {
    "f": {"f", "woman", "girl", "female"},
    "m": {"m", "man", "boy", "male"},
    "trans": {"trans", "trans_f", "trans_m", "transgender"},
}


def _norm(s: str) -> str:
    return re.sub(r"[\s_]", "", str(s).lower())


def _tag_parts(tag: str) -> set[str]:
    t = _norm(tag)
    return {t, *[_norm(p) for p in re.split(r"[-_]", str(tag)) if p]}


def _matches(group: str, tags: list[str]) -> bool:
    """Matches the group against the tag only (answer_info[...][1]); the answer text
    must not be used ("22-year-old" would contain "old")."""
    g = _norm(group)
    cands = _SYN.get(g, {g})
    return bool(cands & _tag_parts(tags[1]))


def bbq_target_idx(r: dict) -> int | None:
    """Index of the answer naming the stereotyped group; None if ambiguous or not found."""
    groups = r["additional_metadata"]["stereotyped_groups"]
    cand = [i for i in range(3) if r["answer_info"][f"ans{i}"][1] != "unknown"]
    # 1) answer_info tags
    hits = [i for i in cand if any(_matches(g, r["answer_info"][f"ans{i}"]) for g in groups)]
    if len(hits) == 1:
        return hits[0]
    # 2) answer text, whole word (e.g. Nationality: tag = region, text = nationality)
    def in_text(g, i):
        txt = f"{r['answer_info'][f'ans{i}'][0]} {r[f'ans{i}']}".lower()
        return re.search(rf"(?<![\w-]){re.escape(g.lower())}(?![\w-])", txt) is not None
    hits = [i for i in cand if any(in_text(g, i) for g in groups)]
    return hits[0] if len(hits) == 1 else None


def bbq_unknown_idx(r: dict) -> int:
    return next(i for i in range(3) if r["answer_info"][f"ans{i}"][1] == "unknown")


def load_bbq() -> pd.DataFrame:
    rows = []
    for c in BBQ_CATS:
        with open(BBQ_DIR / f"{c}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                unk = bbq_unknown_idx(r)
                tgt = bbq_target_idx(r)
                other = next((i for i in range(3) if i not in (unk, tgt)), None) if tgt is not None else None
                stereo = (tgt if r["question_polarity"] == "neg" else other) if tgt is not None else None
                rows.append({
                    "uid": f"bbq:{c}:{r['example_id']}",
                    "category": c,
                    "template": f"{c}:{r['question_index']}",
                    "context_condition": r["context_condition"],
                    "question_polarity": r["question_polarity"],
                    "context": r["context"], "question": r["question"],
                    "ans0": r["ans0"], "ans1": r["ans1"], "ans2": r["ans2"],
                    "label": int(r["label"]), "unknown_idx": unk,
                    "target_idx": tgt, "stereo_idx": stereo,
                })
    df = pd.DataFrame(rows)
    df["target_idx"] = df["target_idx"].astype("Int64")
    df["stereo_idx"] = df["stereo_idx"].astype("Int64")
    return df


def _ss_split_prefix(context: str, sentence: str) -> tuple[str, str]:
    """Intrasentence: context = text before BLANK, completion = rest of the filled-in sentence."""
    prefix = context.split("BLANK")[0].rstrip()
    if prefix and sentence.lower().startswith(prefix.lower()):
        return sentence[:len(prefix)], sentence[len(prefix):].strip()
    return "", sentence.strip()


def load_stereoset() -> pd.DataFrame:
    rows = []
    for kind in ["intrasentence", "intersentence"]:
        df = pd.read_parquet(DATA / f"stereoset_{kind}.parquet")
        for _, e in df.iterrows():
            sents = e["sentences"]
            rec = {"uid": f"ss:{kind}:{e['id']}", "kind": kind, "bias_type": e["bias_type"],
                   "target": e["target"], "raw_context": e["context"]}
            lab = {0: "anti", 1: "stereo", 2: "unrelated"}
            for s, g in zip(sents["sentence"], sents["gold_label"]):
                if kind == "intrasentence":
                    ctx, comp = _ss_split_prefix(e["context"], s)
                else:
                    ctx, comp = e["context"].strip(), s.strip()
                rec[f"ctx_{lab[int(g)]}"] = ctx
                rec[f"comp_{lab[int(g)]}"] = comp
            rows.append(rec)
    out = pd.DataFrame(rows)
    # context shared by the three options (for intrasentence the prefix can differ only in case)
    out["context"] = out["ctx_stereo"]
    return out


def load_mmlu(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"mmlu_{split}.parquet")
    if split == "auxiliary_train":
        df = df.rename(columns=lambda c: c.replace("train.", ""))
    df = df.reset_index(drop=True)
    df["uid"] = [f"mmlu:{split}:{i}" for i in range(len(df))]
    return df


def _assign(keys: list[str], rng: random.Random) -> dict[str, str]:
    keys = sorted(set(keys))
    rng.shuffle(keys)
    n = len(keys)
    n_tr = max(1, round(n * SPLIT_FRACS["train"]))
    n_dev = max(1, round(n * SPLIT_FRACS["dev"]))
    out = {}
    for i, k in enumerate(keys):
        out[k] = "train" if i < n_tr else ("dev" if i < n_tr + n_dev else "test")
    return out


def make_splits(force: bool = False) -> dict:
    if SPLITS_FILE.exists() and not force:
        return json.loads(SPLITS_FILE.read_text())
    rng = random.Random(SEED)
    bbq = load_bbq()
    bbq_map = {}
    for c in BBQ_CATS:
        bbq_map.update(_assign(bbq.loc[bbq.category == c, "template"].tolist(), rng))
    ss = load_stereoset()
    ss["key"] = ss.bias_type + ":" + ss.target
    ss_map = {}
    for c in SS_CATS:
        ss_map.update(_assign(ss.loc[ss.bias_type == c, "key"].tolist(), rng))
    splits = {"seed": SEED, "fracs": SPLIT_FRACS, "bbq_template": bbq_map, "ss_target": ss_map}
    SPLITS_FILE.write_text(json.dumps(splits, indent=1))
    return splits


def with_splits(bbq: pd.DataFrame | None = None, ss: pd.DataFrame | None = None):
    sp = make_splits()
    if bbq is not None:
        bbq = bbq.copy(); bbq["split"] = bbq.template.map(sp["bbq_template"])
    if ss is not None:
        ss = ss.copy(); ss["split"] = (ss.bias_type + ":" + ss.target).map(sp["ss_target"])
    return bbq, ss


def mmlu_dev_subset(n_per_subject: int = 20) -> pd.DataFrame:
    """MMLU dev subset from the validation split (stratified by subject)."""
    v = load_mmlu("validation")
    return v.groupby("subject", group_keys=False).apply(lambda g: g.head(n_per_subject)).reset_index(drop=True)


def mmlu_neutral_pool(n: int = 4000) -> pd.DataFrame:
    """Neutral examples for the BAD probe: a sample of auxiliary_train (never used for evaluation).
    As in FairSteer (App. C.1): items whose answer is D are excluded and option D is removed, so the
    3-option format matches BBQ and the classifier cannot separate them by format."""
    a = load_mmlu("auxiliary_train")
    a = a[a.answer != 3].copy()
    a["choices"] = a.choices.apply(lambda c: list(c)[:3])
    return a.sample(n=n, random_state=SEED).reset_index(drop=True)
