"""One-off check of how Kokoro/misaki reads lettered options (not part of the build)."""
import modal
from modal_build import KOKORO_REPO, KOKORO_REV, VOL, tts_image, volume

app = modal.App("sb-spoken-letters")


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=900)
def run():
    import numpy as np
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    from scipy.signal import resample_poly
    from transformers import pipeline as hf_pipeline
    cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    q = "Where are you likely to find water underneath?"
    ch = ["toilet", "sink", "jar", "bridge", "house"]
    L = "ABCDE"
    variants = {
        "plain_dot": q + " " + " ".join(f"{l}. {c}." for l, c in zip(L, ch)),
        "plain_comma": q + " " + " ".join(f"{l}, {c}." for l, c in zip(L, ch)),
        "option_word": q + " " + " ".join(f"Option {l}: {c}." for l, c in zip(L, ch)),
        "newline": q + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(L, ch)),
    }
    for v in ("af_heart", "bm_george", "am_eric"):
        vp = hf_hub_download(KOKORO_REPO, f"voices/{v}.pt", revision=KOKORO_REV)
        p = pipes["b" if v.startswith("b") else "a"]
        for name, t in variants.items():
            res = list(p(t, voice=vp, speed=1.0))
            ph = " | ".join(r.phonemes for r in res)
            x = np.concatenate([r.audio.cpu().numpy() for r in res if r.audio is not None])
            hyp = asr({"raw": resample_poly(x, 2, 3).astype("float32"), "sampling_rate": 16000})["text"]
            print("VAR", v, name, "|", ph[-160:], "|| ASR:", hyp, flush=True)


@app.local_entrypoint()
def main():
    run.remote()
