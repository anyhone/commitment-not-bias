"""Steering hooks (CAA / FairSteer) and activation capture.

Sign convention (the same throughout the project): the DSV points TOWARDS the bias
(mean(biased) − mean(unbiased)) and is added with a negative coefficient `coef`.

Positions: each batch row gets `start` (first modified token) and `cls_pos`
(token read by the BAD classifier).
- CAA (Rimsky et al.): all positions start <= t < length are modified (after the instruction).
- FairSteer (Li et al., Eq. 4): only the activation of the last prompt token (cls_pos) is modified,
  at the chosen layer only; the change propagates to the following layers.
"""
from __future__ import annotations

import torch

from .model import get_layers, hidden_of


class Steerer:
    def __init__(self, model, layer: int):
        self.layer = layer
        self.mode = "off"          # off | caa | fairsteer
        self.coef = 0.0
        self.vec = None            # [B, d] (one vector per row) or [d]
        self.start = self.cls_pos = self.length = None
        self.probe = None          # dict(mean, scale, w, b) of fp32 tensors on GPU; output = P(biased)
        self.threshold = 0.5
        self.last_only = False     # True for FairSteer: modify cls_pos only
        self.last_prob = self.last_flag = None
        self.handle = get_layers(model)[layer].register_forward_hook(self._hook)

    def set_batch(self, start, cls_pos, length, vec):
        dev = vec.device if torch.is_tensor(vec) else torch.device("cuda")
        self.start = torch.as_tensor(start, device=dev)
        self.cls_pos = torch.as_tensor(cls_pos, device=dev)
        self.length = torch.as_tensor(length, device=dev)
        self.vec = vec

    def _hook(self, module, inputs, output):
        if self.mode == "off":
            self.last_flag = self.last_prob = None
            return output
        h = hidden_of(output)
        B, T, _ = h.shape
        rows = torch.arange(B, device=h.device)
        if self.mode == "fairsteer":
            x = h[rows, self.cls_pos].float()
            p = self.probe
            z = ((x - p["mean"]) / p["scale"]) @ p["w"] + p["b"]
            prob = torch.sigmoid(z)
            flag = prob > self.threshold
            self.last_prob = prob.detach().cpu()
        else:
            flag = torch.ones(B, dtype=torch.bool, device=h.device)
            self.last_prob = None
        self.last_flag = flag.detach().cpu()
        pos = torch.arange(T, device=h.device)[None, :]
        if self.last_only:
            m = (pos == self.cls_pos[:, None]) & flag[:, None]
        else:
            m = (pos >= self.start[:, None]) & (pos < self.length[:, None]) & flag[:, None]
        v = self.vec if self.vec.dim() == 2 else self.vec[None, :].expand(B, -1)
        h2 = h + (self.coef * m[..., None].to(h.dtype)) * v[:, None, :].to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    def remove(self):
        self.handle.remove()


class Capturer:
    """Captures, at the requested layers, the activation at one position per row (`pos`) or the
    mean over a span [span_start, span_end)."""

    def __init__(self, model, layers: list[int]):
        self.layers = layers
        self.pos = self.span = None
        self.acts: dict[int, torch.Tensor] = {}
        self.handles = [get_layers(model)[l].register_forward_hook(self._mk(l)) for l in layers]

    def _mk(self, l):
        def hook(module, inputs, output):
            if self.pos is None and self.span is None:  # inactive
                return
            h = hidden_of(output)
            if self.span is not None:
                s, e = self.span
                T = h.shape[1]
                pos = torch.arange(T, device=h.device)[None, :]
                m = ((pos >= s[:, None]) & (pos < e[:, None])).to(h.dtype)[..., None]
                a = (h * m).sum(1) / m.sum(1).clamp_min(1)
            else:
                a = h[torch.arange(h.shape[0], device=h.device), self.pos]
            self.acts[l] = a.detach().to(torch.float16).cpu()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
