"""
Build the base data: train/eval manifests (JSONL) + 16 kHz audio banks.

Runs inside Modal (see training/modal_train.py::prepare), writing to /vol/data:
  train.jsonl          shuffled training items (CREMA-D + MInDS-14 audio, BANKING77 text)
  eval_items.jsonl     every evaluation item with its FIXED option order (CAL, T1..T6, *_text)
  audio/crema_d.npy    all CREMA-D clips, 16 kHz float32, concatenated (+ crema_d.json index)
  audio/minds14.npy    all MInDS-14 English clips, resampled 8 -> 16 kHz (+ minds14.json index)
  data_summary.json    counts, seeds, licences, example prompts

Can also be run by hand: python training/prepare_data.py --out /tmp/base_data
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

import common as C

# --- sources (pinned revisions so the data is reproducible) -------------------------------
CREMA_REPO, CREMA_REV = "cfahlgren1/crema-d", "b7d5beded69285eb5eb6a1c6894a5b0ad17cab92"
MINDS_REPO, MINDS_REV = "PolyAI/minds14", "40ce77cb32a384e4d50a568e1ec39ac804019d33"
MINDS_CONFIGS = ["en-US", "en-GB", "en-AU"]
BANKING_REPO, BANKING_REV = "legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec"
VOTES_URL = ("https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/"
             "1658cd342dff90010aa843eaeebd53610a08b1dc/processedResults/tabulatedVotes.csv")
LICENCES = {
    "CREMA-D": "Open Database License (ODbL) + Database Contents License, "
               "https://github.com/CheyneyComputerScience/CREMA-D (HF mirror cfahlgren1/crema-d)",
    "MInDS-14": "CC BY 4.0, https://huggingface.co/datasets/PolyAI/minds14",
    "BANKING77": "CC BY 4.0, https://huggingface.co/datasets/legacy-datasets/banking77",
    # evaluation-only text guardrail sets (never trained on)
    "CLINC150": "CC BY 3.0, https://huggingface.co/datasets/clinc/clinc_oos (plus/test)",
    "BoolQ": "CC BY-SA 3.0, https://huggingface.co/datasets/google/boolq (validation)",
    "HellaSwag": "MIT (per the authors' GitHub; no HF licence tag), https://huggingface.co/datasets/Rowan/hellaswag (validation)",
    "WinoGrande": "no HF licence tag (eval only), https://huggingface.co/datasets/allenai/winogrande (debiased/validation)",
    "JADE-Data": "Apache-2.0, https://huggingface.co/datasets/theunnecessarythings/JADE-Data (train-only, opt-in)",
}

# Out-of-domain text guardrail sets (evaluation only), fetched through the Hub's parquet export.
# set -> (repo, config, split)
GUARD_SOURCES = {
    "G1": ("clinc/clinc_oos", "plus", "test"),
    "G2": ("google/boolq", "default", "validation"),
    "G3": ("Rowan/hellaswag", "default", "validation"),
    "G4": ("allenai/winogrande", "winogrande_debiased", "validation"),
}
GUARD_N_ITEMS = 500

# JADE-Data: synthetic text choice / yes-no decisions (Apache-2.0), stored apart from train.jsonl. The training
# runs use the larger sample built by modal_train.py::jade instead.
JADE_URL = ("https://huggingface.co/datasets/theunnecessarythings/JADE-Data/resolve/"
            "ca75bcb94afd60c4e9c0c17133da4c731e527ea2/train.jsonl")
JADE_N_ITEMS = 6000
JADE_MAX_INPUT_TOKENS = 1500  # per-row input_tokens as reported by JADE

# --- split / sampling settings ----------------------------------------------------------------
N_TEST_ACTORS = 21  # of 91 CREMA-D actors; the other 70 are "train actors"
N_CALIB_ACTORS = 5  # carved out of the 70 train actors; used only to fit the temperature
MINDS_TEST_FRAC = 0.20  # stratified by intent
MINDS_CALIB_FRAC = 0.10  # of the MInDS-14 train split, kept for temperature fitting
BANKING_TRAIN_FRACTION = 0.45  # BANKING77 text items as a share of ALL training items
BANKING_N_OPTIONS = 20  # gold + 19 random distractors
T5_N_ITEMS = 500
BANKING_N_CAL = 300  # unused BANKING77 train sentences for temperature fitting
DATA_VERSION = 5  # bump when the item builders change, so `prepare` rebuilds the Volume copy
# T6 = the T5 sentences with all 77 intents as options (training uses at most 20, so this tests the untrained
# head rows U onwards)
LABEL_SMOOTHING = 0.05
VOTE_EPS = 0.01
MIN_SAMPLES = 1600  # skip clips shorter than 0.1 s (broken files)

# --- question wording -------------------------------------------------------------------------
CREMA_QUESTIONS = [  # paraphrases for training; T1/T3/CAL always use the first one
    "Which emotion does the speaker express?",
    "How does the speaker sound?",
    "What emotion can you hear in the speaker's voice?",
    "Which option best describes the speaker's emotional tone?",
]
ANGRY_QUESTION = "Does the speaker sound angry?"  # T2 only; no yes/no emotion items in training
MINDS_QUESTION = "What does the caller want to do?"
MINDS_NOUL_TEMPLATE = "Is the caller asking to {desc}?"
BANKING_QUESTION = "What is the customer asking about?"
YES_NO = ["yes", "no"]

MINDS_INTENT_DESCRIPTIONS = {  # human-readable versions of the 14 MInDS-14 class names
    "abroad": "use their card or account abroad",
    "address": "change their address",
    "app_error": "report a problem with the banking app",
    "atm_limit": "ask about their ATM withdrawal limit",
    "balance": "check their account balance",
    "business_loan": "ask about a business loan",
    "card_issues": "report a problem with their card",
    "cash_deposit": "deposit cash",
    "direct_debit": "set up or ask about a direct debit",
    "freeze": "freeze their card or account",
    "high_value_payment": "make a high-value payment",
    "joint_account": "set up or ask about a joint account",
    "latest_transactions": "check their latest transactions",
    "pay_bill": "pay a bill",
}


def banking_description(name: str) -> str:
    """BANKING77 class name -> option text, e.g. 'card_payment_not_recognised' -> 'card payment not recognised'."""
    return name.replace("_", " ").rstrip("?").strip().lower()


# =============================================================================================
# Downloading + decoding
# =============================================================================================


def download_sources(raw_dir: Path) -> dict:
    """Download the parquet files + votes CSV. Returns raw rows and label names."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    def fetch(repo, rev, patterns):
        local = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=patterns,
                                  local_dir=str(raw_dir / repo.replace("/", "__")))
        # only the files this call asked for: the folder is shared across calls for the same repo
        return sorted({f for pat in patterns for f in Path(local).glob(pat)})

    def label_names(table, column):
        meta = json.loads(table.schema.metadata[b"huggingface"])
        return meta["info"]["features"][column]["names"]

    crema_files = fetch(CREMA_REPO, CREMA_REV, ["data/*.parquet"])
    crema_rows = [r for f in crema_files for r in pq.read_table(f).to_pylist()]

    minds_rows, minds_names = [], None
    for cfg in MINDS_CONFIGS:
        for f in fetch(MINDS_REPO, MINDS_REV, [f"{cfg}/*.parquet"]):
            t = pq.read_table(f)
            minds_names = label_names(t, "intent_class")
            minds_rows += t.to_pylist()
    n_paths = len({r["path"] for r in minds_rows})
    if n_paths != len(minds_rows):
        raise RuntimeError(f"MInDS-14: {len(minds_rows)} rows but only {n_paths} unique clips (duplicated download)")
    print(f"[prepare] MInDS-14 English: {len(minds_rows)} clips "
          f"{dict(Counter(r['path'].split('~')[0] for r in minds_rows))}")

    bank_files = fetch(BANKING_REPO, BANKING_REV, ["data/*.parquet"])
    bank_train = pq.read_table([f for f in bank_files if "train" in f.name][0])
    bank_test = pq.read_table([f for f in bank_files if "test" in f.name][0])

    with urllib.request.urlopen(VOTES_URL, timeout=120) as resp:
        votes_csv = resp.read().decode("utf-8")

    jade_rows = []
    with urllib.request.urlopen(JADE_URL, timeout=1800) as resp:
        for line in resp:
            r = json.loads(line)
            if r.get("source", {}).get("input_tokens", 10**9) <= JADE_MAX_INPUT_TOKENS:
                jade_rows.append(r)

    guard = {}
    for set_name, (repo, cfg, split) in GUARD_SOURCES.items():
        api = f"https://huggingface.co/api/datasets/{repo}/parquet/{cfg}/{split}"
        with urllib.request.urlopen(api, timeout=120) as resp:
            urls = json.loads(resp.read())
        tables = []
        for url in urls:
            with urllib.request.urlopen(url, timeout=600) as resp:
                tables.append(pq.read_table(io.BytesIO(resp.read())))
        rows = [r for t in tables for r in t.to_pylist()]
        names = label_names(tables[0], "intent") if set_name == "G1" else None
        guard[set_name] = {"rows": rows, "names": names}

    return {
        "crema_rows": crema_rows,
        "votes": C.parse_voice_votes(votes_csv),
        "minds_rows": minds_rows,
        "minds_names": minds_names,
        "banking_train": bank_train.to_pylist(),
        "banking_test": bank_test.to_pylist(),
        "banking_names": label_names(bank_train, "label"),
        "guard": guard,
        "jade": jade_rows,
    }


def decode_audio(audio_cell: dict) -> np.ndarray:
    """Parquet audio cell {"bytes", "path"} -> 16 kHz mono float32, at most 30 s."""
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # mono
    x = C.resample(x, sr, C.SAMPLE_RATE)
    return x[: C.MAX_AUDIO_SECONDS * C.SAMPLE_RATE].astype(np.float32)


# =============================================================================================
# Item builders
# =============================================================================================


def split_crema_actors(actors: list[str], seed: int = C.SEED) -> dict[str, str]:
    """91 actors -> 21 test / 70 train (seed 0); 5 of the 70 become the calibration split."""
    order = sorted(actors)
    random.Random(seed).shuffle(order)
    test, rest = order[:N_TEST_ACTORS], order[N_TEST_ACTORS:]
    calib, train = rest[:N_CALIB_ACTORS], rest[N_CALIB_ACTORS:]
    return {**{a: "test" for a in test}, **{a: "calib" for a in calib}, **{a: "train" for a in train}}


def build_crema(rows, votes, decode=decode_audio):
    """CREMA-D -> (train items, eval items, audio clips). Split by actor, never by clip."""
    actors = sorted({key.split("_")[0] for key in votes})
    split = split_crema_actors(actors)
    rng = random.Random(f"{C.SEED}:crema")
    train, evals, clips, skipped = [], [], {}, Counter()
    test_rows = []
    for r in sorted(rows, key=lambda r: r["source_file"]):
        key = r["source_file"].removesuffix(".wav")
        if key not in votes:
            skipped["no_votes"] += 1
            continue
        x = decode(r["audio"])
        if len(x) < MIN_SAMPLES:
            skipped["too_short"] += 1
            continue
        clips[key] = x
        counts = votes[key]
        dist = C.votes_to_distribution(counts, VOTE_EPS)
        majority = [i for i, c in enumerate(counts) if c == counts.max()]
        gold = C.EMOTIONS.index(C.EMOTION_CODES[r["emotion_code"]])
        meta = {"actor": r["actor_id"], "emotion": r["emotion_code"], "votes": counts.tolist()}
        common_args = dict(source="crema_d", options=C.EMOTIONS, target=dist, gold=gold, majority=majority,
                           audio={"bank": "crema_d", "key": key}, meta=meta)
        part = split[key.split("_")[0]]
        if part == "train":
            train.append(C.make_item(item_id=f"train/crema_d/{key}", set_name="train", rng=rng,
                                     question=rng.choice(CREMA_QUESTIONS), **common_args))
        elif part == "calib":
            evals.append(C.make_item(item_id=f"CAL/crema_d/{key}", set_name="CAL", rng=rng,
                                     question=CREMA_QUESTIONS[0], **common_args))
        else:
            test_rows.append((key, r, common_args))

    # Test actors: T1 (clean), T3 (same items over a phone line), T2 (yes/no angry), text controls.
    random.Random(f"{C.SEED}:crema_test_order").shuffle(test_rows)
    for key, r, common_args in test_rows:
        t1 = C.make_item(item_id=f"T1/crema_d/{key}", set_name="T1", rng=rng, question=CREMA_QUESTIONS[0], **common_args)
        t3 = {**t1, "id": f"T3/crema_d/{key}", "set": "T3", "channel": "phone"}
        t1_text = {**t1, "id": f"T1_text/crema_d/{key}", "set": "T1_text", "audio": None, "state_text": r["sentence"]}
        angry = common_args["meta"]["emotion"] == "ANG"
        anger_share = float(common_args["meta"]["votes"][0] / max(1.0, sum(common_args["meta"]["votes"])))
        t2 = C.make_item(item_id=f"T2/crema_d/{key}", set_name="T2", rng=rng, source="crema_d",
                         question=ANGRY_QUESTION, options=YES_NO, target=[1.0, 0.0] if angry else [0.0, 1.0],
                         gold=0 if angry else 1, audio={"bank": "crema_d", "key": key},
                         meta={**common_args["meta"], "anger_vote_share": anger_share})
        t2_text = {**t2, "id": f"T2_text/crema_d/{key}", "set": "T2_text", "audio": None, "state_text": r["sentence"]}
        evals += [t1, t2, t3, t1_text, t2_text]
    stats = {"actors": {p: sorted(a for a, s in split.items() if s == p) for p in ("train", "calib", "test")},
             "skipped": dict(skipped)}
    return train, evals, clips, stats


def _balanced_flags(n: int, rng: random.Random) -> list[bool]:
    flags = [i % 2 == 0 for i in range(n)]  # exactly half True
    rng.shuffle(flags)
    return flags


def build_minds(rows, intent_names, decode=decode_audio):
    """MInDS-14 English -> (train items, eval items, audio clips). Stratified 80/20 by intent;
    10% of the 80% is held out for temperature fitting (CAL)."""
    descs = [MINDS_INTENT_DESCRIPTIONS[n] for n in intent_names]
    k = len(descs)
    rows = sorted(rows, key=lambda r: r["path"])
    split_rng = random.Random(C.SEED)
    part_of = {}
    for intent in range(k):
        idx = [i for i, r in enumerate(rows) if r["intent_class"] == intent]
        split_rng.shuffle(idx)
        n_test = round(MINDS_TEST_FRAC * len(idx))
        rest = idx[n_test:]
        n_cal = round(MINDS_CALIB_FRAC * len(rest))
        part_of.update({i: "test" for i in idx[:n_test]})
        part_of.update({i: "calib" for i in rest[:n_cal]})
        part_of.update({i: "train" for i in rest[n_cal:]})

    rng = random.Random(f"{C.SEED}:minds")
    train, evals, clips = [], [], {}
    order = list(range(len(rows)))
    random.Random(f"{C.SEED}:minds_order").shuffle(order)  # random order -> smoke subsets are mixed
    noul_rows = [i for i in order if part_of[i] in ("train", "calib")]
    noul_true = dict(zip(noul_rows, _balanced_flags(len(noul_rows), rng)))
    for i in order:
        r, part = rows[i], part_of[i]
        key = r["path"]
        clips[key] = decode(r["audio"])
        gold = r["intent_class"]
        audio = {"bank": "minds14", "key": key}
        meta = {"intent": intent_names[gold], "lang": key.split("~")[0]}
        hard = C.smooth_one_hot(k, gold, 0.0)
        if part == "test":
            t4 = C.make_item(item_id=f"T4/minds14/{key}", set_name="T4", source="minds14", question=MINDS_QUESTION,
                             options=descs, target=hard, gold=gold, rng=rng, audio=audio, meta=meta)
            t4_text = {**t4, "id": f"T4_text/minds14/{key}", "set": "T4_text", "audio": None,
                       "state_text": r["transcription"]}
            evals += [t4, t4_text]
            continue
        set_name = "train" if part == "train" else "CAL"
        eps = LABEL_SMOOTHING if part == "train" else 0.0  # CAL is scored against the hard truth
        choice = C.make_item(item_id=f"{set_name}/minds14/choice/{key}", set_name=set_name, source="minds14",
                             question=MINDS_QUESTION, options=descs, target=C.smooth_one_hot(k, gold, eps),
                             gold=gold, rng=rng, audio=audio, meta=meta)
        asked = gold if noul_true[i] else rng.choice([c for c in range(k) if c != gold])
        yes_no_gold = 0 if asked == gold else 1
        noul = C.make_item(item_id=f"{set_name}/minds14/noul/{key}", set_name=set_name, source="minds14",
                           question=MINDS_NOUL_TEMPLATE.format(desc=descs[asked]), options=YES_NO,
                           target=C.smooth_one_hot(2, yes_no_gold, eps), gold=yes_no_gold, rng=rng, audio=audio,
                           meta={**meta, "asked_intent": intent_names[asked]})
        (train if part == "train" else evals).extend([choice, noul])
    return train, evals, clips


def _banking_item(set_name, idx, row, descs, rng, eps, n_options=BANKING_N_OPTIONS):
    gold = row["label"]
    distractors = rng.sample([c for c in range(len(descs)) if c != gold], n_options - 1)
    labels = [gold] + distractors
    return C.make_item(item_id=f"{set_name}/banking77/{idx}", set_name=set_name, source="banking77",
                       question=BANKING_QUESTION, options=[descs[c] for c in labels],
                       target=C.smooth_one_hot(len(labels), 0, eps), gold=0, rng=rng,
                       state_text=row["text"].strip(), meta={"label": row["label"]})


def build_banking(train_rows, test_rows, label_names, n_train):
    """BANKING77 text -> n_train training items + T5 (500 fixed test items, 20 options each)
    + T6 (the same 500 sentences with all 77 options)."""
    descs = [banking_description(n) for n in label_names]
    rng = random.Random(f"{C.SEED}:banking")
    train_idx = sorted(rng.sample(range(len(train_rows)), min(n_train, len(train_rows))))
    train = [_banking_item("train", i, train_rows[i], descs, rng, LABEL_SMOOTHING) for i in train_idx]
    test_idx = rng.sample(range(len(test_rows)), T5_N_ITEMS)
    evals = [_banking_item("T5", i, test_rows[i], descs, rng, 0.0) for i in test_idx]
    rng_cal = random.Random(f"{C.SEED}:banking_cal")  # separate streams: T5 and train stay unchanged
    unused = sorted(set(range(len(train_rows))) - set(train_idx))
    cal_idx = rng_cal.sample(unused, min(BANKING_N_CAL, len(unused)))
    evals += [_banking_item("CAL", i, train_rows[i], descs, rng_cal, 0.0) for i in cal_idx]
    rng_all = random.Random(f"{C.SEED}:banking_all_options")
    evals += [_banking_item("T6", i, test_rows[i], descs, rng_all, 0.0, n_options=len(descs)) for i in test_idx]
    return train, evals


def _clinc_description(name: str) -> str:
    return "something else (none of the listed requests)" if name == "oos" else name.replace("_", " ")


def build_guard(guard: dict) -> list[dict]:
    """Out-of-domain text guardrail items (evaluation only), GUARD_N_ITEMS per set, fixed option order."""
    evals = []
    for set_name in sorted(guard):
        rows, names = guard[set_name]["rows"], guard[set_name]["names"]
        rng = random.Random(f"{C.SEED}:guard:{set_name}")
        idx = sorted(rng.sample(range(len(rows)), min(GUARD_N_ITEMS, len(rows))))
        for i in idx:
            r = rows[i]
            if set_name == "G1":  # CLINC150: 150 intents + out-of-scope, all 151 options
                src, question, state = "clinc150", "What is the user asking for?", r["text"]
                options, gold = [_clinc_description(n) for n in names], int(r["intent"])
            elif set_name == "G2":  # BoolQ: yes/no reading comprehension
                q = r["question"].strip()
                src, question, state = "boolq", q[:1].upper() + q[1:] + "?", r["passage"]
                options, gold = YES_NO, 0 if r["answer"] else 1
            elif set_name == "G3":  # HellaSwag: 4 candidate endings
                src, question, state = "hellaswag", "Which ending most plausibly continues the text?", r["ctx"]
                options, gold = [e.strip() for e in r["endings"]], int(r["label"])
            else:  # G4 WinoGrande: 2 candidates for the blank
                src, question, state = ("winogrande", "In this sentence, who or what does the blank (_) refer to?",
                                        r["sentence"])
                options, gold = [r["option1"], r["option2"]], int(r["answer"]) - 1
            evals.append(C.make_item(item_id=f"{set_name}/{src}/{i}", set_name=set_name, source=src,
                                     question=question, options=options,
                                     target=C.smooth_one_hot(len(options), gold, 0.0), gold=gold, rng=rng,
                                     state_text=state))
    return evals


def build_jade(rows: list[dict]) -> list[dict]:
    """JADE-Data rows -> our training items (seeded sample of JADE_N_ITEMS, label smoothing like other text)."""
    rng = random.Random(f"{C.SEED}:jade")
    picked = rng.sample(range(len(rows)), min(JADE_N_ITEMS, len(rows)))
    items = []
    for i in picked:
        r = rows[i]
        q = r["question"]
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            question = q["instructions"] + (f" (yes = {crit['true']}; no = {crit['false']})"
                                             if "true" in crit and "false" in crit else "")
            options, gold = YES_NO, 0 if r["target"] is True else 1
        else:
            codes = sorted(q["criteria"])
            if not 2 <= len(codes) <= C.N_ANSWER_CODES or r["target"] not in codes:
                continue
            question, options, gold = q["instructions"], [q["criteria"][c] for c in codes], codes.index(r["target"])
        items.append(C.make_item(item_id=f"train/jade/{r['id']}", set_name="train", source="jade",
                                 question=question, options=options,
                                 target=C.smooth_one_hot(len(options), gold, LABEL_SMOOTHING), gold=gold,
                                 rng=rng, state_text=state, meta={"suite": r["suite"]}))
    return items


def build_all(src: dict, decode=decode_audio) -> tuple[list, list, dict, dict]:
    crema_train, crema_eval, crema_clips, crema_stats = build_crema(src["crema_rows"], src["votes"], decode)
    minds_train, minds_eval, minds_clips = build_minds(src["minds_rows"], src["minds_names"], decode)
    n_audio = len(crema_train) + len(minds_train)
    n_text = round(BANKING_TRAIN_FRACTION / (1 - BANKING_TRAIN_FRACTION) * n_audio)
    bank_train, bank_eval = build_banking(src["banking_train"], src["banking_test"], src["banking_names"], n_text)

    train = crema_train + minds_train + bank_train
    random.Random(f"{C.SEED}:train_order").shuffle(train)
    evals = crema_eval + minds_eval + bank_eval + (build_guard(src["guard"]) if src.get("guard") else [])
    # interleave the calibration sources (so smoke subsets of CAL contain all three), then group by set;
    # the sort is stable, so every other set keeps its seeded order
    cal = [it for it in evals if it["set"] == "CAL"]
    random.Random(f"{C.SEED}:cal_order").shuffle(cal)
    evals = cal + [it for it in evals if it["set"] != "CAL"]
    evals.sort(key=lambda it: C.ALL_EVAL_SETS.index(it["set"]))
    clips = {"crema_d": crema_clips, "minds14": minds_clips}
    check_no_leakage(train, evals)
    crema_stats["jade_items"] = build_jade(src["jade"]) if src.get("jade") else []
    return train, evals, clips, crema_stats


def check_no_leakage(train: list[dict], evals: list[dict]) -> None:
    """Raise if train, CAL and test items share a recording, a CREMA-D actor or a BANKING77 sentence."""
    def fingerprints(items):
        out = set()
        for it in items:
            if it.get("audio"):
                key = it["audio"]["key"]
                out.add(("clip", it["audio"]["bank"], key))
                if it["audio"]["bank"] == "crema_d":
                    out.add(("actor", key.split("_")[0]))
            elif it["source"] == "banking77":
                out.add(("text", it["state_text"]))
        return out

    tr = fingerprints(train)
    cal = fingerprints([it for it in evals if it["set"] == "CAL"])
    test = fingerprints([it for it in evals if it["set"] != "CAL" and not it["set"].endswith("_text")])
    for name, a, b in [("train/test", tr, test), ("train/CAL", tr, cal), ("CAL/test", cal, test)]:
        shared = a & b
        if shared:
            raise RuntimeError(f"data leakage {name}: {len(shared)} shared, e.g. {sorted(shared)[:3]}")
    print(f"[prepare] leakage check passed: train / CAL / test share no clip, actor or sentence")


def summarise(train, evals, crema_stats) -> dict:
    def examples():
        out = []
        for source in ["crema_d", "minds14", "banking77"]:
            it = next(i for i in train if i["source"] == source)
            out.append({"id": it["id"], "prompt": C.build_prompt_text(it),
                        "target": {C.ANSWER_CODES[j]: p for j, p in enumerate(it["target"])}})
        return out

    return {
        "data_version": DATA_VERSION,
        "seed": C.SEED,
        "train_items_by_source": dict(Counter(i["source"] for i in train)),
        "train_items_total": len(train),
        "banking_share_of_train": round(sum(i["source"] == "banking77" for i in train) / len(train), 3),
        "eval_items_by_set": dict(Counter(i["set"] for i in evals)),
        "crema_actor_split": crema_stats["actors"],
        "crema_skipped": crema_stats["skipped"],
        "licences": LICENCES,
        "examples": examples(),
    }


def main(out_dir: str, force: bool = False) -> dict:
    out = Path(out_dir)
    stale = (out / "data_summary.json").exists() and \
        json.loads((out / "data_summary.json").read_text()).get("data_version") != DATA_VERSION
    if (out / "data_summary.json").exists() and not force and not stale:
        print(f"[prepare] {out} already prepared; pass --force to rebuild")
        return json.loads((out / "data_summary.json").read_text())
    t0 = time.time()
    print("[prepare] downloading sources ...")
    src = download_sources(Path("/tmp/raw_sources"))
    print(f"[prepare] downloaded in {time.time() - t0:.0f}s; building items ...")
    train, evals, clips, crema_stats = build_all(src)
    for name, bank in clips.items():
        C.write_audio_bank(out / "audio", name, bank)
    C.write_jsonl(out / "train.jsonl", train)
    C.write_jsonl(out / "eval_items.jsonl", evals)
    jade = crema_stats.pop("jade_items", [])
    C.write_jsonl(out / "jade_items.jsonl", jade)  # not used by the training runs (they read sb_jade)
    print(f"[prepare] JADE-Data practice items: {len(jade)} {dict(Counter(j['meta']['suite'] for j in jade).most_common(8))}")
    summary = summarise(train, evals, crema_stats)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[prepare] done in {time.time() - t0:.0f}s: {len(train)} train items, {len(evals)} eval items")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(json.dumps(main(args.out, args.force), indent=2))
