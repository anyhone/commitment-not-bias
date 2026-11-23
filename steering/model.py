"""4-bit model loading and prompt utilities."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

V2 = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    return yaml.safe_load((V2 / "configs" / f"{name}.yaml").read_text())


def load_model(cfg: dict):
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right padding: in causal models the pads do not affect the real tokens
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg.get("revision"),
                                                 quantization_config=bnb, device_map={"": 0},
                                                 dtype=torch.float16)
    model.eval()
    return model, tok


def get_layers(model):
    for path in ["model.layers", "model.language_model.layers", "language_model.model.layers"]:
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("decoder layers not found")


def hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


def chat_prompt(tok, cfg: dict, user_text: str, assistant_prefix: str = "") -> tuple[str, int]:
    """Returns (full text, character index where the user message ends)."""
    msgs = [{"role": "user", "content": user_text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                **cfg.get("template_kwargs", {}))
    end_user = s.rfind(user_text) + len(user_text)
    return s + assistant_prefix, end_user


# ---- multiple-choice prompts, identical for all models --------------------------------------
LETTERS = "ABCD"


def bbq_user_text(r) -> str:
    return (f"{r['context']} {r['question']}\n"
            f"A. {r['ans0']}\nB. {r['ans1']}\nC. {r['ans2']}\n"
            "Answer with only the letter (A, B, or C) of the correct option.")


def mmlu_user_text(r) -> str:
    ch = list(r["choices"])
    subj = str(r.get("subject", "")).replace("_", " ")
    head = f"The following is a multiple choice question about {subj}.\n\n" if subj else ""
    opts = ", ".join(LETTERS[:len(ch) - 1]) + f", or {LETTERS[len(ch) - 1]}"
    return (f"{head}{r['question']}\n" + "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(ch)) +
            f"Answer with only the letter ({opts}) of the correct option.")


def letter_token_ids(tok, n: int) -> list[list[int]]:
    """For each letter, the ids of the single tokens that represent it ("A", " A", ...)."""
    out = []
    for L in LETTERS[:n]:
        ids = set()
        for v in [L, " " + L]:
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
            elif len(enc) == 2 and tok.decode(enc[:1]).strip() == "":  # e.g. ['▁', 'A']
                ids.add(enc[1])
        assert ids, f"no token for letter {L}"
        out.append(sorted(ids))
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
