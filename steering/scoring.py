"""Log-probability scoring, batched with right padding.

- mc_score: multiple choice (BBQ, MMLU). One forward pass; the prediction is the argmax of the
  letter-token logits at the last prompt position. Steering starts at the last token of the
  user message; the classifier reads the last prompt token.
- ll_score: mean log-likelihood of the completion given the context (StereoSet).
  Steering and classification start at the last context token.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .model import chat_prompt, letter_token_ids


def _pad(seqs: list[list[int]], pad_id: int):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids, att


def _batches(lengths, bs):
    order = np.argsort(-np.asarray(lengths), kind="stable")
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def encode_mc(tok, cfg, user_texts: list[str], prefixes: list[str] | None = None):
    enc = []
    for j, t in enumerate(user_texts):
        s, end_user = chat_prompt(tok, cfg, t, prefixes[j] if prefixes else "")
        ids = tok(s, add_special_tokens=False).input_ids
        n_user = len(tok(s[:end_user], add_special_tokens=False).input_ids)
        enc.append((ids, max(0, n_user - 1)))
    return enc


@torch.no_grad()
def mc_score(model, tok, cfg, user_texts, n_opts, steerer=None, vecs=None, capturer=None,
             batch_size=8, desc="mc", prefixes=None):
    """vecs: None, a tensor [d] or a list of tensors [d] (one per item). Returns a dict with
    logprob [N, n_opts], pred [N], flag [N] (FairSteer), acts {layer: [N, d]}."""
    enc = encode_mc(tok, cfg, user_texts, prefixes)
    opt_ids = letter_token_ids(tok, n_opts)
    N = len(enc)
    lp = np.zeros((N, n_opts), dtype=np.float32)
    flags = np.full(N, np.nan, dtype=np.float32)
    probs = np.full(N, np.nan, dtype=np.float32)
    acts = {l: None for l in (capturer.layers if capturer else [])}
    dev = model.device
    for idx in tqdm(list(_batches([len(e[0]) for e in enc], batch_size)), desc=desc, leave=False):
        seqs = [enc[i][0] for i in idx]
        ids, att = _pad(seqs, tok.pad_token_id)
        ids, att = ids.to(dev), att.to(dev)
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        last = lens - 1
        if steerer is not None:
            v = _row_vecs(vecs, idx, dev)
            steerer.set_batch([enc[i][1] for i in idx], last, lens, v)
        if capturer is not None:
            capturer.pos, capturer.span = last, None
        logits = model(input_ids=ids, attention_mask=att).logits
        lg = logits[torch.arange(len(idx), device=dev), last].float()
        logp = torch.log_softmax(lg, -1)
        opt = torch.stack([torch.logsumexp(logp[:, o], -1) for o in opt_ids], -1)
        opt = torch.log_softmax(opt, -1)
        lp[idx] = opt.cpu().numpy()
        if steerer is not None and steerer.last_flag is not None:
            flags[idx] = steerer.last_flag.numpy().astype(np.float32)
            if steerer.last_prob is not None:
                probs[idx] = steerer.last_prob.numpy()
        if capturer is not None:
            for l in capturer.layers:
                if acts[l] is None:
                    acts[l] = torch.zeros((N, capturer.acts[l].shape[-1]), dtype=torch.float16)
                acts[l][torch.as_tensor(idx)] = capturer.acts[l]
    if capturer is not None:
        capturer.pos = capturer.span = None
    return {"logprob": lp, "pred": lp.argmax(1), "flag": flags, "prob_biased": probs, "acts": acts}


def _row_vecs(vecs, idx, dev):
    if vecs is None:
        return None
    if torch.is_tensor(vecs):
        return vecs.to(dev)
    return torch.stack([vecs[i] for i in idx]).to(dev)


def encode_ll(tok, pairs: list[tuple[str, str]]):
    """pairs: (context, completion). The context is tokenized with special tokens (BOS);
    the completion is preceded by a space."""
    enc, mismatch = [], 0
    for ctx, comp in pairs:
        ctx = ctx.rstrip()
        full = (ctx + " " + comp.lstrip()) if ctx else comp.strip()
        c_ids = tok(ctx, add_special_tokens=True).input_ids if ctx else tok("", add_special_tokens=True).input_ids
        f_ids = tok(full, add_special_tokens=True).input_ids
        n_ctx = max(1, len(c_ids))
        if f_ids[:len(c_ids)] != c_ids:
            mismatch += 1
        enc.append((f_ids, n_ctx))
    return enc, mismatch


@torch.no_grad()
def ll_score(model, tok, pairs, steerer=None, vecs=None, capturer=None, batch_size=16, desc="ll",
             capture_at="span"):
    """capture_at: 'span' = mean over the completion tokens (DSV extraction),
    'cls' = last context token (position read by the BAD classifier)."""
    enc, mismatch = encode_ll(tok, pairs)
    N = len(enc)
    out = np.zeros(N, dtype=np.float64)
    flags = np.full(N, np.nan, dtype=np.float32)
    acts = {l: None for l in (capturer.layers if capturer else [])}
    dev = model.device
    for idx in tqdm(list(_batches([len(e[0]) for e in enc], batch_size)), desc=desc, leave=False):
        seqs = [enc[i][0] for i in idx]
        nctx = torch.tensor([enc[i][1] for i in idx], device=dev)
        ids, att = _pad(seqs, tok.pad_token_id)
        ids, att = ids.to(dev), att.to(dev)
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        if steerer is not None:
            steerer.set_batch(nctx - 1, nctx - 1, lens, _row_vecs(vecs, idx, dev))
        if capturer is not None:
            if capture_at == "span":
                capturer.pos, capturer.span = None, (nctx, lens)
            else:
                capturer.pos, capturer.span = nctx - 1, None
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ids[:, 1:]
        tok_lp = logp.gather(-1, tgt[..., None])[..., 0]          # lp of token t+1 given <=t
        pos = torch.arange(tgt.shape[1], device=dev)[None, :] + 1  # position of the predicted token
        m = (pos >= nctx[:, None]) & (pos < lens[:, None])
        s = (tok_lp * m).sum(1) / m.sum(1).clamp_min(1)
        out[idx] = s.cpu().numpy()
        if steerer is not None and steerer.last_flag is not None:
            flags[idx] = steerer.last_flag.numpy().astype(np.float32)
        if capturer is not None:
            for l in capturer.layers:
                if acts[l] is None:
                    acts[l] = torch.zeros((N, capturer.acts[l].shape[-1]), dtype=torch.float16)
                acts[l][torch.as_tensor(idx)] = capturer.acts[l]
    if capturer is not None:
        capturer.pos = capturer.span = None
    return {"ll": out, "flag": flags, "acts": acts, "tokenization_mismatch": mismatch}
