"""
The decision head: hidden state at the answer position -> one probability per option.

Recipe follows Perplexity's pplx-decider-v1.1:
  * readout = Linear(hidden_size, 255, bias=False), one row per answer code (A, B, ..., AA, AB, ...)
  * initialised by copying the language-model output rows of those code tokens, so at initialisation the
    head gives the model's own next-token logits for the codes (the `stock` baseline = head at init)
  * last-token pooling: the final hidden state at the answer position
  * logits beyond the item's N options are masked, then softmax over the N valid codes
  * loss = soft-target cross-entropy (+ optional Brier term, weight 0 by default)

`softcap` reproduces a model's final logit soft-cap (c * tanh(logits / c)) when it has one; None for Qwen3-Omni.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, n_codes: int = 255, softcap: float | None = None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, n_codes, bias=False)  # kept in float32
        self.softcap = softcap

    @classmethod
    def from_lm_head(cls, lm_head_weight: torch.Tensor, code_token_ids: list[int], softcap: float | None):
        """Copy the output-embedding rows of the answer-code tokens into a new head."""
        head = cls(lm_head_weight.shape[1], len(code_token_ids), softcap)
        with torch.no_grad():
            ids = torch.tensor(code_token_ids, device=lm_head_weight.device)
            head.proj.weight.copy_(lm_head_weight[ids].float().to(head.proj.weight.device))
        return head

    def forward(self, hidden: torch.Tensor, n_options) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden: (batch, hidden_size) last-token states. n_options: number of options per row.
        Returns (log_probs (batch, 255) with -inf beyond each row's N, valid mask)."""
        logits = self.proj(hidden.float())
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        n = torch.as_tensor(n_options, device=logits.device)
        valid = torch.arange(logits.shape[-1], device=logits.device)[None, :] < n[:, None]
        logits = logits.masked_fill(~valid, float("-inf"))
        return torch.log_softmax(logits, dim=-1), valid

    def save(self, path, codes: list[str]) -> None:
        from safetensors.torch import save_file

        meta = {"softcap": json.dumps(self.softcap), "codes": json.dumps(codes)}
        save_file({"weight": self.proj.weight.detach().float().cpu().contiguous()}, str(path), metadata=meta)

    @classmethod
    def load(cls, path, device="cpu") -> "DecisionHead":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")
            softcap = json.loads(f.metadata()["softcap"])
        head = cls(weight.shape[1], weight.shape[0], softcap)
        with torch.no_grad():
            head.proj.weight.copy_(weight)
        return head.to(device)


def init_head(model, code_token_ids: list[int]) -> DecisionHead:
    """Head at initialisation for a model whose text config has final_logit_softcapping (PEFT-wrapped too)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    softcap = base.config.get_text_config().final_logit_softcapping
    device = next(base.parameters()).device
    return DecisionHead.from_lm_head(base.get_output_embeddings().weight, code_token_ids, softcap).to(device)


def last_hidden_states(model, enc) -> torch.Tensor:
    """Run the multimodal decoder (with LoRA if present) and return the final hidden state at the
    last position of every row, shape (batch, hidden_size). The LM head is skipped."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(base.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    attn = enc["attention_mask"]
    position_ids = attn.long().cumsum(-1) - 1  # positions that ignore the left padding,
    position_ids = position_ids.masked_fill(attn == 0, 1)  # same convention as model.generate()
    out = base.model(**enc, position_ids=position_ids, use_cache=False)
    return out.last_hidden_state[:, -1, :]


def decision_log_probs(model, head: DecisionHead, enc, n_options) -> tuple[torch.Tensor, torch.Tensor]:
    """Full readout: processor output -> (log_probs over codes, valid mask)."""
    return head(last_hidden_states(model, enc), n_options)


def decision_loss(log_probs, valid, targets, brier_weight: float = 0.0) -> torch.Tensor:
    """Soft-target cross-entropy (= KL(target || model) + const) + optional Brier term.
    `targets` is (batch, 255), zero beyond each row's N options."""
    safe = log_probs.masked_fill(~valid, 0.0)  # avoid 0 * -inf = nan
    loss = -(targets * safe).sum(-1)
    if brier_weight:
        probs = log_probs.exp()  # exactly 0 at masked codes
        loss = loss + brier_weight * ((probs - targets) ** 2).sum(-1)
    return loss.mean()


def check_answer_codes(tokenizer, codes_file) -> list[int]:
    """Rebuild the code list from the live tokenizer and make sure it matches answer_codes.json."""
    import common

    saved = json.loads(Path(codes_file).read_text())
    codes, ids = common.build_answer_codes(tokenizer, saved["n"])
    if codes != saved["codes"] or ids != saved["token_ids"]:
        raise RuntimeError("answer codes from the tokenizer differ from answer_codes.json — regenerate it")
    return ids


def enable_bidirectional(thinker) -> None:
    """Lift the causal mask in Qwen3-Omni's text decoder (as in pplx-decider): every prompt token attends to every
    other non-padding token. Only for one-pass scoring (no generation, no KV cache); applies to this process."""
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as qom
    from transformers.masking_utils import create_bidirectional_mask

    def bidirectional_mask(config, inputs_embeds, attention_mask, past_key_values=None, position_ids=None, **_):
        return create_bidirectional_mask(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                         past_key_values=past_key_values, allow_is_bidirectional_skip=False)

    qom.create_causal_mask = bidirectional_mask
    for layer in thinker.model.layers:  # SDPA would otherwise fall back to is_causal=True when given no mask
        layer.self_attn.is_causal = False
