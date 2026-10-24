"""Synthesis back-ends for sb_synthetic, run in Modal (one image per family). Each takes jobs ({key, text, voice,
params} for TTS, {key, src_key} for copy-synthesis) and returns ({key: 16 kHz float32 array}, fails). Pinned repos,
revisions and licences are in REPOS."""
from __future__ import annotations

import math
import time

import numpy as np

SR = 16000

REPOS = {
    "kokoro": {"weights": "hexgrad/Kokoro-82M", "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
               "weights_licence": "apache-2.0 (HF cardData)", "code": "pip kokoro==0.9.4 (github hexgrad/kokoro, Apache-2.0)",
               "g2p": "misaki[en]==0.9.4 (Apache-2.0; espeak-ng GPL-3 binary used only as OOV fallback phonemizer)"},
    "piper": {"weights": "rhasspy/piper-voices", "revision": "5041a7a49ae5b3a0f30e6837769eb03fdbb231bd",
              "weights_licence": "mit (HF cardData); per-voice MODEL_CARD datasets: libritts CC BY 4.0, ljspeech / kristin / "
                                 "norman / john / cori = public domain (LibriVox / LJ Speech); only voices trained from scratch or "
                                 "fine-tuned from a public-domain voice (no lessac / ryan / NC / SA lineage)",
              "code": "pip piper-tts==1.2.0 (github rhasspy/piper, MIT; not the GPL piper1 >= 1.3) + piper-phonemize==1.1.0 (MIT; "
                      "bundles espeak-ng data, GPL-3, as phonemizer)"},
    "speecht5": {"weights": "microsoft/speecht5_tts", "revision": "30fcde30f19b87502b8435427b5f5068e401d5f6",
                 "weights_licence": "mit (HF cardData)", "vocoder": "microsoft/speecht5_hifigan@bb6f429406e86a9992357a972c0698b22043307d (mit)",
                 "speaker_embeddings": "Matthijs/cmu-arctic-xvectors@5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5 (mit; CMU ARCTIC)",
                 "code": "transformers==4.46.3 (Apache-2.0)"},
    "parler": {"weights": "parler-tts/parler-tts-mini-v1", "revision": "0392b9451a601e528fd863bbb0598431fee810d9",
               "weights_licence": "apache-2.0 (HF cardData; trained on MLS-eng + LibriTTS-R, CC BY 4.0)",
               "code": "github huggingface/parler-tts@d108732cd57788ec86bc857d99a6cabd66663d68 (Apache-2.0)"},
    "chatterbox": {"weights": "ResembleAI/chatterbox", "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
                   "weights_licence": "mit (HF cardData)", "code": "pip chatterbox-tts==0.1.2 (github resemble-ai/chatterbox, MIT); "
                   "outputs carry Resemble's Perth imperceptible watermark (resemble-perth, MIT)"},
    "vocos": {"weights": "charactr/vocos-mel-24khz", "revision": "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
              "weights_licence": "mit (HF cardData)", "code": "pip vocos==0.1.0 (github gemelo-ai/vocos, MIT)"},
    "hifigan": {"weights": "microsoft/speecht5_hifigan", "revision": "bb6f429406e86a9992357a972c0698b22043307d",
                "weights_licence": "mit (HF cardData)", "code": "transformers==4.46.3 (Apache-2.0); features = SpeechT5 log-mel"},
}

KOKORO_TRAIN = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_river",
                "af_sarah", "af_sky", "am_adam", "am_echo", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                "bf_alice", "bf_emma", "bf_isabella", "bm_daniel", "bm_fable", "bm_george"]
KOKORO_CAL = ["af_nova", "am_eric", "bf_lily", "bm_lewis"]

# piper voice -> path inside rhasspy/piper-voices (all checked MODEL_CARDs: permissive data, no lessac/ryan lineage)
PIPER_VOICES = {
    "en_US-libritts-high": "en/en_US/libritts/high/en_US-libritts-high",
    "en_US-ljspeech-high": "en/en_US/ljspeech/high/en_US-ljspeech-high",
    "en_US-kristin-medium": "en/en_US/kristin/medium/en_US-kristin-medium",
    "en_US-norman-medium": "en/en_US/norman/medium/en_US-norman-medium",
    "en_GB-cori-high": "en/en_GB/cori/high/en_GB-cori-high",
    "en_US-john-medium": "en/en_US/john/medium/en_US-john-medium",
}
PIPER_LIBRITTS_TRAIN_SPK = list(range(0, 900, 23))  # 40 of the 904 libritts-high speakers
PIPER_LIBRITTS_CAL_SPK = [5, 300, 611, 877]
PIPER_SINGLE_TRAIN = ["en_US-ljspeech-high", "en_US-kristin-medium", "en_US-norman-medium", "en_GB-cori-high"]
PIPER_SINGLE_CAL = ["en_US-john-medium"]

SPEECHT5_TRAIN_SPK = ["awb", "bdl", "clb", "jmk", "ksp", "slt"]
SPEECHT5_CAL_SPK = ["rms"]

PARLER_NAMES_TRAIN = ["Laura", "Gary", "Jon", "Lea", "Karen", "Rick", "Brenda", "David", "Eileen", "Jordan", "Mike",
                      "Yann", "Joy", "James", "Eric", "Lauren", "Rose", "Will", "Jason", "Aaron", "Naomie", "Alisa",
                      "Patrick", "Jerry", "Tina", "Jenna", "Bill", "Carol", "Barbara", "Rebecca"]
PARLER_NAMES_CAL = ["Anna", "Bruce", "Emily", "Tom"]


def to16k(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr == SR:
        return x.astype(np.float32)
    g = math.gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def _ok(y: np.ndarray, text: str | None) -> str | None:
    """Reject empty / silent / runaway outputs (None = ok)."""
    if y is None or len(y) < int(0.6 * SR) or not np.isfinite(y).all():
        return "short_or_nan"
    if float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) < 1e-4:
        return "silent"
    if text:
        exp = max(1.0, len(text) / 14.0)  # ~14 chars/s of English speech
        d = len(y) / SR
        if d > 3.0 * exp + 2.0 or d < 0.25 * exp:
            return f"duration_off({d:.1f}s vs ~{exp:.1f}s)"
    return None


# Kokoro
def run_kokoro(jobs: list[dict]):
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    import torch

    r = REPOS["kokoro"]
    cfg = hf_hub_download(r["weights"], "config.json", revision=r["revision"])
    mdl = hf_hub_download(r["weights"], "kokoro-v1_0.pth", revision=r["revision"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(config=cfg, model=mdl).to(dev).eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    vp, out, fails = {}, {}, []
    for j in jobs:
        v = j["voice"]
        try:
            if v not in vp:
                vp[v] = hf_hub_download(r["weights"], f"voices/{v}.pt", revision=r["revision"])
            pipe = pipes["b" if v.startswith("b") else "a"]
            ch = [c.audio.detach().cpu().numpy() for c in pipe(j["text"], voice=vp[v], speed=j["params"]["speed"])
                  if c.audio is not None]
            y = to16k(np.concatenate(ch), 24000)
            bad = _ok(y, j["text"])
            if bad:
                raise ValueError(bad)
            out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails.append([j["key"], repr(e)[:200]])
    return out, fails


# Piper
def run_piper(jobs: list[dict]):
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    r = REPOS["piper"]
    voices, out, fails = {}, {}, []
    for j in jobs:
        vname = (j.get("params") or {}).get("voice_file") or j["voice"]
        try:
            if vname not in voices:
                base = PIPER_VOICES[vname]
                onnx = hf_hub_download(r["weights"], base + ".onnx", revision=r["revision"])
                cfg = hf_hub_download(r["weights"], base + ".onnx.json", revision=r["revision"])
                voices[vname] = PiperVoice.load(onnx, config_path=cfg, use_cuda=False)
            pv = voices[vname]
            p = j["params"]
            raw = b"".join(pv.synthesize_stream_raw(j["text"], speaker_id=p.get("speaker_id"),
                                                    length_scale=p["length_scale"], noise_scale=p["noise_scale"],
                                                    noise_w=p["noise_w"], sentence_silence=0.2))
            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            y = to16k(y, pv.config.sample_rate)
            bad = _ok(y, j["text"])
            if bad:
                raise ValueError(bad)
            out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails.append([j["key"], repr(e)[:200]])
    return out, fails


# SpeechT5 (+ copy-synthesis)
def _xvectors():
    import io
    import zipfile

    from huggingface_hub import hf_hub_download

    z = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset",
                        revision="5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5")
    by = {}
    with zipfile.ZipFile(z) as f:
        for n in sorted(f.namelist()):
            if not n.endswith(".npy"):
                continue
            stem = n.rsplit("/", 1)[-1]
            spk = stem.split("_")[2] if stem.startswith("cmu_us_") else None
            if spk:
                by.setdefault(spk, []).append(np.load(io.BytesIO(f.read(n))).astype(np.float32).reshape(-1))
    return {k: np.stack(v) for k, v in by.items()}


def run_speecht5(jobs: list[dict]):
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    torch.set_num_threads(4)
    r = REPOS["speecht5"]
    proc = SpeechT5Processor.from_pretrained(r["weights"], revision=r["revision"])
    model = SpeechT5ForTextToSpeech.from_pretrained(r["weights"], revision=r["revision"]).eval()
    voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", revision=REPOS["hifigan"]["revision"]).eval()
    xv = _xvectors()
    out, fails = {}, []
    for j in jobs:
        try:
            p = j["params"]
            rs = np.random.default_rng(p["seed"])
            vecs = []
            for spk, w in p["mix"]:
                pool = xv[spk]
                vecs.append(w * pool[rs.integers(len(pool), size=3)].mean(0))
            e = np.sum(vecs, axis=0)
            e = e / (np.linalg.norm(e) + 1e-9)
            ids = proc(text=j["text"], return_tensors="pt")["input_ids"][:, :580]
            with torch.inference_mode():
                y = model.generate_speech(ids, torch.tensor(e)[None], vocoder=voc).numpy()
            y = y.astype(np.float32)
            bad = _ok(y, j["text"])
            if bad:
                raise ValueError(bad)
            out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails.append([j["key"], repr(e)[:200]])
    return out, fails


def _pool():
    import json
    from pathlib import Path

    d = Path("/vol/stageb_work/sb_synthetic/pool")
    data = np.load(d / "real_pool.npy", mmap_mode="r")
    idx = json.loads((d / "real_pool.json").read_text())
    return lambda k: np.array(data[idx[k][0]: idx[k][0] + idx[k][1]], dtype=np.float32)


def run_hifigan(jobs: list[dict]):
    """Copy-synthesis: SpeechT5 log-mel of the real 16 kHz clip -> SpeechT5 HiFi-GAN."""
    import torch
    from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan

    torch.set_num_threads(4)
    fe = SpeechT5FeatureExtractor.from_pretrained(REPOS["speecht5"]["weights"], revision=REPOS["speecht5"]["revision"])
    voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", revision=REPOS["hifigan"]["revision"]).eval()
    get = _pool()
    out, fails = {}, []
    for j in jobs:
        try:
            x = get(j["src_key"])
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            mel = fe(audio_target=x, sampling_rate=SR, return_tensors="pt")["input_values"]
            with torch.inference_mode():
                y = voc(mel[0]).numpy().astype(np.float32)
            y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
            bad = _ok(y, None)
            if bad:
                raise ValueError(bad)
            out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails.append([j["key"], repr(e)[:200]])
    return out, fails


def run_vocos(jobs: list[dict]):
    """Copy-synthesis: Vocos mel (24 kHz) of the real clip -> Vocos decoder -> 16 kHz."""
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from vocos import Vocos

    torch.set_num_threads(4)
    r = REPOS["vocos"]
    cfg = hf_hub_download(r["weights"], "config.yaml", revision=r["revision"])
    ckpt = hf_hub_download(r["weights"], "pytorch_model.bin", revision=r["revision"])
    voc = Vocos.from_hparams(cfg)
    voc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    voc.eval()
    get = _pool()
    out, fails = {}, []
    from scipy.signal import resample_poly
    for j in jobs:
        try:
            x = get(j["src_key"])
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            x24 = resample_poly(x.astype(np.float64), 3, 2).astype(np.float32)
            with torch.inference_mode():
                mel = voc.feature_extractor(torch.from_numpy(x24)[None])
                y = voc.decode(mel)[0].numpy()
            y = to16k(y, 24000)
            y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
            bad = _ok(y, None)
            if bad:
                raise ValueError(bad)
            out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails.append([j["key"], repr(e)[:200]])
    return out, fails


# Parler-TTS (GPU)
def run_parler(jobs: list[dict], batch: int = 16):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    r = REPOS["parler"]
    model = ParlerTTSForConditionalGeneration.from_pretrained(r["weights"], revision=r["revision"],
                                                              torch_dtype=torch.bfloat16).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(r["weights"], revision=r["revision"])
    sr = model.config.sampling_rate
    out, fails = {}, []
    jobs = sorted(jobs, key=lambda j: len(j["text"]))
    for b0 in range(0, len(jobs), batch):
        bj = jobs[b0: b0 + batch]
        try:
            torch.manual_seed(bj[0]["params"]["seed"])
            d = tok([j["params"]["description"] for j in bj], return_tensors="pt", padding=True).to("cuda")
            p = tok([j["text"] for j in bj], return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                g = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                   do_sample=True, temperature=1.0, return_dict_in_generate=True,
                                   max_new_tokens=int(86 * (max(len(j["text"]) for j in bj) / 11.0 + 2.5)))
            for i, j in enumerate(bj):
                a = g.sequences[i, : g.audios_length[i]].float().cpu().numpy()
                y = to16k(a, sr)
                bad = _ok(y, j["text"])
                if bad:
                    fails.append([j["key"], bad])
                else:
                    out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails += [[j["key"], repr(e)[:200]] for j in bj]
        print("parler batch", b0 // batch, "of", (len(jobs) + batch - 1) // batch, "ok", len(out), flush=True)
    return out, fails


# Chatterbox (GPU)
def run_chatterbox(jobs: list[dict], time_budget_s: float = 1e9):
    import tempfile
    from pathlib import Path

    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    from chatterbox.tts import ChatterboxTTS

    r = REPOS["chatterbox"]
    ck = snapshot_download(r["weights"], revision=r["revision"],
                           allow_patterns=["ve.pt", "ve.safetensors", "t3_cfg.safetensors", "s3gen.pt", "s3gen.safetensors",
                                           "tokenizer.json", "conds.pt"])
    m = ChatterboxTTS.from_local(ck, "cuda")
    get = _pool()
    tmp = Path(tempfile.mkdtemp())
    out, fails, t0 = {}, [], time.time()
    for j in jobs:
        if time.time() - t0 > time_budget_s:
            fails.append([j["key"], "time_budget"])
            continue
        try:
            p = j["params"]
            ref = tmp / f"{p['ref_key']}.wav"
            if not ref.exists():
                sf.write(str(ref), get(p["ref_key"]), SR)
            torch.manual_seed(p["seed"])
            with torch.inference_mode():
                w = m.generate(j["text"], audio_prompt_path=str(ref), exaggeration=p["exaggeration"],
                               cfg_weight=p["cfg_weight"], temperature=p["temperature"])
            y = to16k(w.squeeze(0).float().cpu().numpy(), m.sr)
            bad = _ok(y, j["text"])
            if bad:
                raise ValueError(bad)
            out[j["key"]] = y
        except Exception as e:  # noqa: BLE001
            fails.append([j["key"], repr(e)[:200]])
        if len(out) % 20 == 1:
            print("chatterbox", len(out), "ok", len(fails), "fails", round(time.time() - t0), "s", flush=True)
    return out, fails


RUNNERS = {"kokoro": run_kokoro, "piper": run_piper, "speecht5": run_speecht5, "hifigan": run_hifigan,
           "vocos": run_vocos, "parler": run_parler, "chatterbox": run_chatterbox}
