"""
Decision Index engine (kit `Engine` subclass) for homi: Qwen3-Omni-30B-A3B thinker, optionally with the
trained LoRA adapter merged in, + the decision head (last-token hidden state -> one logit per answer code).

    --engine di_engine:EgmaHomiEngine --option adapter_dir=. --option temperature_file=./temperature.json
    --engine di_engine:EgmaHomiEngine --option system=stock          (untouched Qwen3-Omni, head = lm_head rows)

Each question is rendered with the prompt the adapter was trained with (common.build_prompt_text): state, question,
options as answer codes A), B), ... and the instruction. No truncation, no option pruning, one template for every
benchmark; requests that do not fit raise Unsupported (counted as wrong).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported

import common as C
import jev
import readout

QWEN_OMNI_ID, QWEN_OMNI_REV = "Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"
N_CODES = 255  # the kit allows 2..255 options per choice
TOKEN_BUDGET = 32_768  # padded tokens per forward pass (memory guard); questions of one request are batched


class _LastPositionOnly:
    """Stands in for lm_head (152k-wide logits at every position); the hidden state is read through a hook."""

    def __new__(cls):
        import torch

        class Stub(torch.nn.Module):
            def forward(self, hidden):
                return hidden[:, -1:, :1]

        return Stub()


class EgmaHomiEngine(Engine):
    name = "homi"
    latency = ("Device-synchronized in-process request wall time including prompt construction and tokenisation; "
               "all questions of a request batched into as few forward passes as fit a 32k padded-token budget; "
               "excludes model loading.")

    def __init__(self, system="homi", adapter_dir=None, temperature=1.0, temperature_file=None, max_tokens=None,
                 decision_file=None, **options):
        super().__init__(**options)
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        self.system = system
        if temperature_file and Path(temperature_file).exists():  # the one temperature fitted on the CAL split
            temperature = json.loads(Path(temperature_file).read_text())["temperature"]
        self.temperature = float(temperature)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_OMNI_ID, revision=QWEN_OMNI_REV)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"  # last position of every row = answer position
        full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            QWEN_OMNI_ID, revision=QWEN_OMNI_REV, dtype=torch.bfloat16, device_map="cuda")
        full.disable_talker()
        thinker = full.thinker
        self.codes, code_ids = C.build_answer_codes(self.tok, N_CODES)
        head = readout.DecisionHead.from_lm_head(thinker.lm_head.weight, code_ids, None)  # stock rows, fp32
        trained_rows = 0
        if adapter_dir:
            from peft import PeftModel
            from safetensors import safe_open

            thinker = PeftModel.from_pretrained(thinker, str(adapter_dir)).merge_and_unload()
            path = Path(adapter_dir) / "readout.safetensors"
            with safe_open(str(path), framework="pt", device="cpu") as f:
                weight = f.get_tensor("weight").float()
                trained_codes = json.loads(f.metadata()["codes"])
            if trained_codes != self.codes[: len(trained_codes)]:
                raise ValueError("trained head codes differ from this tokenizer's answer codes")
            trained_rows = weight.shape[0]
            with torch.no_grad():  # trained rows replace the stock ones; rows beyond them stay at lm_head init
                head.proj.weight[:trained_rows].copy_(weight.to(head.proj.weight.device))
        self.head = head.to("cuda").eval()
        self.bidirectional = bool(decision_file and Path(decision_file).exists()
                                  and json.loads(Path(decision_file).read_text()).get("bidirectional"))
        if self.bidirectional:  # the model was trained with the causal mask lifted
            readout.enable_bidirectional(thinker)
        thinker.lm_head = _LastPositionOnly()
        self.model = thinker.eval()
        self.capture = {}
        self.model.model.register_forward_hook(lambda _m, _a, out: self.capture.__setitem__(
            "hidden", (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[:, -1]))
        text_cfg = full.config.thinker_config.text_config
        limit = getattr(text_cfg, "max_position_embeddings", None) or 32768
        self.limit = min(limit, int(max_tokens)) if max_tokens else limit
        self.provenance = {
            "kind": "in-process", "repo": QWEN_OMNI_ID, "revision": QWEN_OMNI_REV, "system": system,
            "adapter_dir": str(adapter_dir) if adapter_dir else None, "trained_head_rows": trained_rows,
            "answer_codes": len(self.codes), "temperature": self.temperature, "context_limit_tokens": self.limit,
            "dtype": "bfloat16", "talker": "disabled", "bidirectional_attention": self.bidirectional,
            "policy": "Each question is one prompt (state, question, options as answer codes); the decision head "
                      "reads the last-token hidden state and gives one logit per code, masked to the question's "
                      "options; probabilities are the softmax over them. No truncation, no option filtering, one "
                      "fixed template; prompts over the context limit are refused as unsupported.",
        }

    def runtime(self):
        import torch
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "device": "cuda"}

    def synchronize(self):
        self.torch.cuda.synchronize()

    # -----------------------------------------------------------------------------------------
    def _render(self, name, q, state_text):
        if q.get("type") not in ("choice", "noul"):
            raise Unsupported("Unsupported question type " + str(q.get("type")))
        q = dict(q)
        if q["type"] == "noul" and not isinstance(q.get("criteria"), dict):
            q["criteria"] = {}
        if q["type"] == "choice" and len(q.get("criteria") or {}) > N_CODES:
            raise Unsupported(f"more than {N_CODES} options")
        item, keys = jev.question_to_item(name, q, False, state_text)
        prompt = C.build_prompt_text(item, self.codes)
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}], add_generation_prompt=True,
            tokenize=False)
        return text, keys

    def _forward(self, texts, n_options):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        self.capture.clear()
        with torch.inference_mode():
            self.model(**enc, use_cache=False)
            log_probs, _ = self.head(self.capture["hidden"], n_options)
        return log_probs.float().cpu()

    def _score(self, texts, lengths, n_options):
        """Pack questions (shortest first) into forward passes under the padded-token budget; on OOM, halve."""
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        out = [None] * len(texts)
        cap = len(texts)
        i = 0
        while i < len(order):
            batch = []
            while i + len(batch) < len(order) and len(batch) < cap:
                j = order[i + len(batch)]
                if batch and lengths[j] * (len(batch) + 1) > TOKEN_BUDGET:
                    break
                batch.append(j)
            try:
                lp = self._forward([texts[j] for j in batch], [n_options[j] for j in batch])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise Unsupported(f"out of memory on one {lengths[batch[0]]}-token prompt")
                cap = max(1, len(batch) // 2)
                continue
            for row, j in zip(lp, batch):
                out[j] = row[: n_options[j]]
            i += len(batch)
        return out

    def __call__(self, state, questions):
        state_text = None if state in ("", None, {}, []) else jev._text(state)
        names, texts, keys_list = [], [], []
        for name, q in questions.items():
            text, keys = self._render(name, q, state_text)
            names.append(name)
            texts.append(text)
            keys_list.append(keys)
        lengths = [len(ids) for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]
        if max(lengths) > self.limit:
            raise Unsupported(f"prompt of {max(lengths)} tokens exceeds the {self.limit}-token context limit")
        rows = self._score(texts, lengths, [len(k) for k in keys_list])
        answers = {}
        for name, keys, row in zip(names, keys_list, rows):
            p = self.torch.softmax(row / self.temperature, -1).tolist()
            s = math.fsum(p)
            p = [x / s for x in p]
            q = questions[name]
            if q["type"] == "noul":  # jev.question_to_item puts "yes" first
                answers[name] = {"type": "noul", "noul": p[0]}
            else:
                probs = dict(zip(keys, p))
                answers[name] = {"type": "choice", "choice": max(keys, key=probs.get), "probabilities": probs}
        return {"model": self.system, "answers": answers,
                "usage": {"input_tokens": int(sum(lengths))}}, None

    def warmup(self):
        warm = {"warmup": {"type": "choice", "instructions": "Which color is named?",
                           "criteria": {"red": "red", "blue": "blue"}}}
        for _ in range(2):
            self("The color is red.", warm)
