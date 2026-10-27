"""
homi: ask typed questions about an audio clip and get one calibrated probability per option.

    python predict.py clip.wav                      # three example questions (yes/no, choice, score)
    python predict.py clip.wav --questions q.json   # your own questions (same shape as EXAMPLE_QUESTIONS)

    from predict import AudioDecider
    decider = AudioDecider()                        # loads the base model + this folder's adapter
    decider.ask("clip.wav", EXAMPLE_QUESTIONS)

Needs one GPU with about 70 GB free (bf16), plus: torch, transformers>=5.18, peft, safetensors, soundfile, scipy, numpy.
This folder holds adapter_model.safetensors + adapter_config.json (LoRA on the Qwen3-Omni thinker), readout.safetensors
(decision head) and temperature.json (one temperature). The base model is downloaded from Hugging Face on first use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import jev  # noqa: E402
import readout  # noqa: E402

MODEL_NAME = "homi"
QWEN_OMNI_ID, QWEN_OMNI_REV = "Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"
EXAMPLE_QUESTIONS = {
    "sounds_angry": {"type": "noul", "instructions": "Does the speaker sound angry?"},
    "emotion": {"type": "choice", "instructions": "Which emotion does the speaker express?",
                "criteria": {"anger": None, "happy": None, "neutral": None, "sad": None}},
    "pace": {"type": "score", "instructions": "How fast is the person speaking?", "criteria": ["slow", "normal pace", "fast"]},
}


class AudioDecider:
    def __init__(self, adapter_dir: str | Path = HERE):
        import torch
        from peft import PeftModel
        from safetensors import safe_open
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        adapter_dir = Path(adapter_dir)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_OMNI_ID, revision=QWEN_OMNI_REV)
        self.processor.tokenizer.padding_side = "left"  # the last position of every row is the answer position
        full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            QWEN_OMNI_ID, revision=QWEN_OMNI_REV, dtype=torch.bfloat16, device_map="cuda")
        full.disable_talker()
        thinker = full.thinker
        self.codes, code_ids = C.build_answer_codes(self.processor.tokenizer, C.N_ANSWER_CODES)
        head = readout.DecisionHead.from_lm_head(thinker.lm_head.weight, code_ids, None)
        thinker = PeftModel.from_pretrained(thinker, str(adapter_dir)).merge_and_unload()
        with safe_open(str(adapter_dir / "readout.safetensors"), framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight").float()
            trained_codes = json.loads(f.metadata()["codes"])
        if trained_codes != self.codes[: len(trained_codes)]:
            raise ValueError("the trained head's answer codes differ from this tokenizer's answer codes")
        with torch.no_grad():  # trained rows replace the stock ones; rows beyond them stay at their lm_head values
            head.proj.weight[: weight.shape[0]].copy_(weight.to(head.proj.weight.device))
        self.head = head.to("cuda").eval()
        self.temperature = json.loads((adapter_dir / "temperature.json").read_text())["temperature"]
        self.capture = {}
        thinker.model.register_forward_hook(lambda _m, _a, out: self.capture.__setitem__(
            "hidden", (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[:, -1]))

        class _LastOnly(torch.nn.Module):  # skip the 152k-wide lm_head; the hook above reads the hidden state
            def forward(self, hidden):
                return hidden[:, -1:, :1]

        thinker.lm_head = _LastOnly()
        self.model = thinker.eval()

    def ask(self, audio, questions: dict, text: str | None = None) -> dict:
        """audio: a file path or a 16 kHz mono float array (or None for text only). questions: {name: {"type":
        "noul" | "choice" | "score", "instructions": ..., "criteria": ...}}. text: optional context shown with the
        audio. All questions run in one batched forward pass. Returns {name: answer}."""
        torch = self.torch
        if isinstance(audio, (str, Path)):
            audio = jev.load_audio_file(audio)
        built = [(name, q, *jev.question_to_item(name, q, audio is not None, text)) for name, q in questions.items()]
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": ([{"type": "audio", "audio": "clip.wav"}] if audio is not None else [])
              + [{"type": "text", "text": C.build_prompt_text(item, self.codes)}]}],
            add_generation_prompt=True, tokenize=False) for _, _, item, _ in built]
        enc = self.processor(text=texts, audio=[audio] * len(built) if audio is not None else None,
                             return_tensors="pt", padding=True)
        enc = {k: v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda") for k, v in enc.items()}
        self.capture.clear()
        with torch.inference_mode():
            self.model(**enc, use_cache=False)
            log_probs, _ = self.head(self.capture["hidden"], [len(item["options"]) for _, _, item, _ in built])
        out = {}
        for (name, q, item, keys), row in zip(built, log_probs.float().cpu().numpy()):
            probs = C.apply_temperature(row[: len(keys)], self.temperature)
            out[name] = jev.probs_to_answer(q["type"], keys, probs, q.get("criteria"))
        return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="predict.py", description=f"{MODEL_NAME}: typed questions about one audio clip, "
                                 "one calibrated probability per option.")
    ap.add_argument("audio", help="audio file (any sample rate; mixed down to 16 kHz mono)")
    ap.add_argument("--questions", help="JSON file {name: {type, instructions, criteria}}; default: three examples")
    ap.add_argument("--text", help="optional text context shown with the audio")
    ap.add_argument("--adapter-dir", default=str(HERE), help="folder with the homi files (default: this folder)")
    args = ap.parse_args(argv)
    questions = json.loads(Path(args.questions).read_text()) if args.questions else EXAMPLE_QUESTIONS
    decider = AudioDecider(args.adapter_dir)
    print(json.dumps(decider.ask(args.audio, questions, args.text), indent=2))


if __name__ == "__main__":
    main()
