"""
The homi training recipe as plain Python (no Modal, no GPU): run configs, builder lists, licence and corpus
exclusions, the data mixture, teacher-label blending, batching and the temperature fit.
Imported by modal_train.py; needs only numpy and inference/common.py on the path.

Modal Volume layout (mounted at /vol):
    /vol/data/            base items from training/prepare_data.py (train.jsonl, eval_items.jsonl, audio/)
    /vol/stageb/<name>/   one folder per data builder in data/ (items_train.jsonl, items_cal.jsonl, audio/)
    /vol/stageb/teacher/  soft_labels.jsonl from data/teacher/
    /vol/runs/<run>/      adapter, logs, temperature, data audit
"""

from __future__ import annotations

import builtins
import json
import math
import re
from pathlib import Path

VOL = "/vol"
DATA_DIR = f"{VOL}/data"
RUNS_DIR = f"{VOL}/runs"
STAGEB_DIR = f"{VOL}/stageb"

LORA_DROPOUT = 0.05  # rank (alpha = 2 x rank) is set per run in RUN_CONFIGS
LR = 1e-4  # LoRA and decision head
WARMUP_FRAC = 0.03
EFFECTIVE_BATCH = 16  # items per optimizer step, unless a run sets "batch"
BRIER_WEIGHT = 0.0  # pure soft-target cross-entropy
TRAIN_TOKEN_BUDGET = 16_384  # max padded tokens per micro-batch (memory guard)
EVAL_TOKEN_BUDGET, EVAL_MAX_BATCH = 32_768, 32
LOG_EVERY = 10
CKPT_EVERY, SMOKE_CKPT_EVERY = 100, 5  # training checkpoints (resume after an interruption)
SMOKE_EVAL_ITEMS, SMOKE_TRAIN_STEPS = 20, 10

# Builder order fixes the seeded shuffles.
PASS1_BUILDERS = ["sb_voice", "sb_sounds", "sb_calls", "sb_spoken", "sb_jade", "sb_tasksource", "sb_sounds2",
                  "sb_relative", "sb_spoken2", "sb_dialog", "sb_calltext", "sb_phone", "sb_aicall", "sb_count",
                  "sb_compare", "sb_phon"]
PASS2_BUILDERS = ["sb_speakers", "sb_synthetic", "sb_intonation", "sb_disfluency", "sb_phone_pitch"]

# One entry per training run; outputs go to /vol/runs/<run>/.
RUN_CONFIGS = {
    # pass 1: from the base model, all pass-1 builders in one run. No MMSU source corpus (VCTK dropped; MELD and
    # RAVDESS never used). Share-alike sources are dropped; CREMA-D (ODbL) is kept, with attribution.
    "pass1": {"backbone": "qwen3-omni", "lora_r": 16, "epochs": 1, "crema_intent_mix": 0.5, "stageb": True,
              "drop_sources": ["banking77"], "batch": 64, "teacher_mix": 0.5, "ordinal_smooth": 0.2,
              "drop_corpora": ["vctk"], "drop_share_alike": True, "mix": "MIX", "builders": PASS1_BUILDERS},
    # pass 2: continues pass 1 at a lower learning rate on the pass-2 builders plus a 20% replay of pass 1's data.
    # The replay leaves out three sources that taught shortcuts (counting speakers by clip length, a too-simple
    # pause-meaning rule).
    "pass2": {"backbone": "qwen3-omni", "lora_r": 16, "epochs": 1, "crema_intent_mix": 0.5, "stageb": True,
              "drop_sources": ["banking77"], "batch": 64, "teacher_mix": 0.5, "ordinal_smooth": 0.2,
              "drop_corpora": ["vctk"], "drop_share_alike": True, "drop_share_alike_backgrounds": True,
              "mix": "MIX_PASS2", "cal_drop_sources": ["banking77"],
              "init_from": f"{RUNS_DIR}/pass1/adapter", "lr": 5e-5, "replay_from": "pass1", "replay_fraction": 0.2,
              "replay_drop_sources": ["sb_count_speakers", "sb_voice_speaker_count", "sb_compare_pause_meaning"],
              "builders": PASS1_BUILDERS + PASS2_BUILDERS},
    # the pass-1 recipe for SMOKE_TRAIN_STEPS steps on 2 GPUs
    "smoke-pass1": {"backbone": "qwen3-omni", "lora_r": 16, "epochs": 1, "crema_intent_mix": 0.5, "stageb": True,
                    "drop_sources": ["banking77"], "batch": 16, "teacher_mix": 0.5, "ordinal_smooth": 0.2,
                    "drop_corpora": ["vctk"], "drop_share_alike": True, "mix": "MIX", "builders": PASS1_BUILDERS},
}

# Share-alike (CC BY-SA) data, dropped when drop_share_alike is set so the weights can be Apache-2.0.
SHARE_ALIKE_SOURCES = {"sb_calltext_e22", "sb_dialog_earnings21"}  # Earnings-22 / Earnings-21 (CC BY-SA 4.0)
SHARE_ALIKE_CORPORA = {"vystadial", "vocalsound"}  # Vystadial (CC BY-SA 3.0), VocalSound (CC BY-SA 4.0)
# per-item flags cover the rest (FMA / OpenMIC tracks in sb_sounds2, tasksource tasks)
SHARE_ALIKE_LICENCE = re.compile(r"by-sa|sharealike|share-alike", re.IGNORECASE)
# MUSAN music files licensed CC BY-SA (from MUSAN's LICENSE files). sb_phone mixes some in as background music; pass 2
# drops those items (drop_share_alike_backgrounds). Pass 1 trained on them.
MUSAN_SHARE_ALIKE = frozenset({
    "music-fma-0021", "music-fma-0022", "music-fma-0023", "music-fma-0024", "music-fma-0025", "music-fma-0026",
    "music-fma-0027", "music-fma-0049", "music-fma-0050", "music-fma-0051", "music-fma-0052", "music-fma-0054",
    "music-fma-0056", "music-fma-0062", "music-fma-0064", "music-fma-0065", "music-fma-0088", "music-fma-0089",
    "music-fma-0090", "music-fma-0092", "music-fma-0093", "music-fma-0105", "music-fma-0106", "music-fma-0107",
    "music-fma-0108", "music-fma-0110", "music-fma-0111", "music-hd-0000", "music-hd-0001", "music-hd-0002",
    "music-hd-0004", "music-hd-0005", "music-hd-0006", "music-hd-0008", "music-hd-0009", "music-hd-0010",
    "music-hd-0011", "music-hd-0012", "music-hd-0013", "music-hd-0014", "music-hd-0015", "music-hd-0016",
    "music-hd-0017", "music-hd-0018", "music-hd-0019", "music-hd-0020", "music-hd-0021", "music-hd-0022",
    "music-hd-0031", "music-hd-0032", "music-hd-0033", "music-hd-0035", "music-hd-0036", "music-hd-0037",
    "music-hd-0038", "music-hd-0049", "music-hd-0050", "music-hd-0051", "music-hd-0053", "music-hd-0054",
    "music-hd-0056", "music-hd-0058", "music-hd-0060", "music-hd-0062", "music-hd-0063", "music-hd-0066",
    "music-hd-0068", "music-hd-0069", "music-hd-0070", "music-hd-0071", "music-hd-0072", "music-hd-0073",
    "music-hd-0074", "music-jamendo-0000", "music-jamendo-0001", "music-jamendo-0002", "music-jamendo-0003",
    "music-jamendo-0004", "music-jamendo-0005", "music-jamendo-0006", "music-jamendo-0007", "music-jamendo-0008",
    "music-jamendo-0018", "music-jamendo-0019", "music-jamendo-0020", "music-jamendo-0021", "music-jamendo-0031",
    "music-jamendo-0032", "music-jamendo-0033", "music-jamendo-0034", "music-jamendo-0036", "music-jamendo-0037",
    "music-jamendo-0040", "music-jamendo-0041", "music-jamendo-0047", "music-jamendo-0048", "music-jamendo-0049",
    "music-jamendo-0050", "music-jamendo-0052", "music-jamendo-0053", "music-jamendo-0054", "music-jamendo-0055",
    "music-jamendo-0057", "music-jamendo-0059", "music-jamendo-0060", "music-jamendo-0061", "music-jamendo-0063",
    "music-jamendo-0064", "music-jamendo-0065", "music-jamendo-0066", "music-jamendo-0067", "music-jamendo-0068",
    "music-jamendo-0071", "music-jamendo-0072", "music-jamendo-0073", "music-jamendo-0074", "music-jamendo-0075",
    "music-jamendo-0077", "music-jamendo-0078", "music-jamendo-0079", "music-jamendo-0080", "music-jamendo-0081",
    "music-jamendo-0082", "music-jamendo-0083", "music-jamendo-0084", "music-jamendo-0085", "music-jamendo-0086",
    "music-jamendo-0088", "music-jamendo-0090", "music-jamendo-0091", "music-jamendo-0092", "music-jamendo-0093",
    "music-jamendo-0094", "music-jamendo-0095", "music-jamendo-0096", "music-jamendo-0097", "music-jamendo-0098",
    "music-jamendo-0099", "music-jamendo-0100", "music-jamendo-0108", "music-jamendo-0109", "music-jamendo-0111",
    "music-jamendo-0112", "music-jamendo-0113", "music-jamendo-0114", "music-jamendo-0115", "music-jamendo-0116",
    "music-jamendo-0118", "music-jamendo-0120", "music-jamendo-0121", "music-jamendo-0122", "music-jamendo-0123",
    "music-jamendo-0124", "music-jamendo-0125", "music-jamendo-0126", "music-jamendo-0127", "music-jamendo-0129",
    "music-jamendo-0131", "music-jamendo-0132", "music-jamendo-0133", "music-jamendo-0142", "music-jamendo-0143",
    "music-jamendo-0144", "music-jamendo-0145", "music-jamendo-0146", "music-jamendo-0154", "music-jamendo-0155",
    "music-jamendo-0156", "music-jamendo-0157", "music-jamendo-0158", "music-jamendo-0159", "music-jamendo-0165",
    "music-jamendo-0166", "music-jamendo-0167", "music-jamendo-0168", "music-jamendo-0169", "music-jamendo-0170",
    "music-jamendo-0171", "music-jamendo-0172", "music-jamendo-0173", "music-jamendo-0174", "music-jamendo-0175",
    "music-jamendo-0176", "music-jamendo-0177", "music-jamendo-0178", "music-jamendo-0179", "music-jamendo-0180",
    "music-jamendo-0181", "music-jamendo-0182", "music-jamendo-0183", "music-jamendo-0184", "music-jamendo-0185",
    "music-jamendo-0186", "music-jamendo-0187", "music-jamendo-0189", "music-jamendo-0190", "music-jamendo-0191",
    "music-jamendo-0193", "music-jamendo-0194", "music-jamendo-0195", "music-jamendo-0198", "music-jamendo-0204",
    "music-jamendo-0205", "music-jamendo-0206", "music-jamendo-0207", "music-jamendo-0208", "music-jamendo-0209",
    "music-jamendo-0210", "music-jamendo-0212", "music-jamendo-0213", "music-jamendo-0214", "music-jamendo-0215",
    "music-jamendo-0216"
})
# Data mixture: (source prefixes, factor), first match wins. factor > 1 repeats items, < 1 keeps a seeded sample;
# unlisted sources 1.0.
MIX = [(("sb_count_", "sb_compare_", "sb_phon_"), 2.0),  # counting, comparison, phonetics
       (("sb_phone_pitch_",), 1.0),  # pass-2 builder; must not fall under "sb_phone_" below
       (("sb_phone_",), 0.8), (("sb_dialog_",), 0.6),
       (("fsd50k", "nsynth", "fma", "openmic"), 0.35),  # sounds and music: the base model is already strong
       (("ts:", "jade"), 0.35)]  # text-only decisions
MIX_PASS2 = [(("sb_phone_pitch_phone_",), 0.75)] + MIX
MIXES = {"MIX": MIX, "MIX_PASS2": MIX_PASS2}


def run_config(run: str) -> dict:
    if run not in RUN_CONFIGS:
        raise ValueError(f"unknown run {run!r}; known: {sorted(RUN_CONFIGS)}")
    return dict(RUN_CONFIGS[run])


def is_smoke(run: str) -> bool:
    return run.startswith("smoke")


def run_dir(run: str) -> Path:
    return Path(RUNS_DIR) / run


def stageb_builders(run: str | None = None) -> list[Path]:
    """The builder folders a run reads: the list frozen in /vol/runs/<run>/stageb_builders.json when training
    started, else the run's configured list, else every finished builder in /vol/stageb."""
    frozen = Path(RUNS_DIR) / run / "stageb_builders.json" if run else None
    if frozen and frozen.exists():
        return [Path(STAGEB_DIR) / b for b in json.loads(frozen.read_text())]
    if run and RUN_CONFIGS.get(run, {}).get("builders"):
        return [Path(STAGEB_DIR) / b for b in RUN_CONFIGS[run]["builders"]]
    return sorted(p for p in Path(STAGEB_DIR).iterdir() if (p / "items_train.jsonl").exists()) \
        if Path(STAGEB_DIR).exists() else []


def freeze_builders(run: str) -> list[str]:
    """Write the run's builder list to /vol/runs/<run>/stageb_builders.json (once) and return it."""
    out = run_dir(run)
    f = out / "stageb_builders.json"
    if not f.exists():
        out.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(run_config(run)["builders"]))
    return json.loads(f.read_text())


def stageb_items(split: str, run: str | None = None) -> list[dict]:
    """All builders' items for `split` ("train" or "cal")."""
    import common

    return [it for b in stageb_builders(run) for it in common.read_jsonl(b / f"items_{split}.jsonl")
            if (b / f"items_{split}.jsonl").exists()]


def all_banks(run: str) -> dict:
    """Base audio banks plus, for runs that use builders, every builder's banks."""
    import common

    banks = common.load_banks(DATA_DIR)
    if run_config(run).get("stageb"):
        for b in stageb_builders(run):
            names = sorted(p.stem for p in (b / "audio").glob("*.npy"))
            banks.update(common.load_banks(b, names))
    return banks


def valid_items(items: list[dict], banks: dict, max_options: int = 40) -> tuple[list[dict], dict]:
    """Drop malformed items instead of failing mid-run; returns (kept, dropped counts by source)."""
    from collections import Counter

    kept, dropped = [], Counter()
    for it in items:
        try:
            k = len(it["options"])
            ok = (2 <= k <= max_options and len(it["target"]) == k and abs(sum(it["target"]) - 1) < 1e-3
                  and 0 <= it["gold"] < k and isinstance(it["question"], str)
                  and builtins.all(isinstance(o, str) and o for o in it["options"])
                  and (it.get("audio") is None or (it["audio"]["bank"] in banks
                                                   and it["audio"]["key"] in banks[it["audio"]["bank"]].index)))
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            it.setdefault("source", "unknown")
            kept.append(it)
        else:
            dropped[it.get("source", "unknown") if isinstance(it, dict) else "?"] += 1
    return kept, dict(dropped)


def is_share_alike(it: dict, backgrounds: bool = False) -> bool:
    m = it.get("meta") or {}
    if backgrounds and isinstance(m.get("background"), dict):
        files = {x for v in m["background"].values() for x in (v if isinstance(v, list) else [v]) if isinstance(x, str)}
        if files & MUSAN_SHARE_ALIKE:
            return True
    return (it.get("source") in SHARE_ALIKE_SOURCES or m.get("corpus") in SHARE_ALIKE_CORPORA
            or m.get("share_alike") is True or bool(SHARE_ALIKE_LICENCE.search(str(m.get("licence") or ""))))


def drop_excluded(items: list[dict], cfg: dict, what: str = "train") -> list[dict]:
    """Remove the sources, corpora and licences a run excludes (applied to training and CAL items)."""
    from collections import Counter

    gone = Counter()
    keep = []
    for it in items:
        corpus = (it.get("meta") or {}).get("corpus")
        if it.get("source") in cfg.get("drop_sources", ()):
            gone[f"source:{it['source']}"] += 1
        elif corpus in cfg.get("drop_corpora", ()):
            gone[f"corpus:{corpus}"] += 1
        elif cfg.get("drop_share_alike") and is_share_alike(it, cfg.get("drop_share_alike_backgrounds", False)):
            gone[f"share-alike:{it.get('source')}"] += 1
        else:
            keep.append(it)
    if gone:
        print(f"[{what}] excluded {sum(gone.values())} items: {dict(gone.most_common())}", flush=True)
    return keep


def apply_mix(items: list[dict], mix: list) -> list[dict]:
    import math
    import random

    import common

    rng = random.Random(f"{common.SEED}:mix")
    out = []
    for it in items:
        f = next((fac for prefixes, fac in mix if it.get("source", "").startswith(prefixes)), 1.0)
        n = math.floor(f) + (1 if rng.random() < f - math.floor(f) else 0)
        out += [it if k == 0 else {**it, "id": f"{it['id']}~r{k}"} for k in range(n)]
    rng.shuffle(out)
    return out


def load_train_items(run: str) -> list[dict]:
    """A run's training items: base train.jsonl plus every builder's items (or, with replay_from, the new builders'
    items plus a seeded share of the earlier run's), minus exclusions, with teacher labels blended and the mixture
    applied."""
    import random

    import common

    items = common.read_jsonl(f"{DATA_DIR}/train.jsonl")
    cfg = run_config(run)
    if cfg.get("drop_sources"):
        items = [it for it in items if it["source"] not in cfg["drop_sources"]]
    if cfg.get("stageb"):
        if cfg.get("replay_from"):
            old_names = {b.name for b in stageb_builders(cfg["replay_from"])}
            rng = random.Random(f"{common.SEED}:replay")
            frac = cfg.get("replay_fraction", 0.3)
            old_items = items + [it for b in stageb_builders(run) if b.name in old_names
                                 for it in common.read_jsonl(b / "items_train.jsonl")]
            if cfg.get("replay_drop_sources"):
                n0 = len(old_items)
                old_items = [it for it in old_items if it.get("source") not in cfg["replay_drop_sources"]]
                print(f"[train] replay: left out {n0 - len(old_items)} items from {cfg['replay_drop_sources']}")
            new_items = [it for b in stageb_builders(run) if b.name not in old_names
                         for it in common.read_jsonl(b / "items_train.jsonl")]
            items = [it for it in old_items if rng.random() < frac] + new_items
            rng.shuffle(items)
            print(f"[train] continued training: {len(new_items)} new items + {len(items) - len(new_items)} replayed "
                  f"({frac:.0%} of {len(old_items)})")
        else:
            items += stageb_items("train", run)
            random.Random(f"{common.SEED}:stageb").shuffle(items)
        if cfg.get("drop_corpora") or cfg.get("drop_share_alike"):
            items = drop_excluded(items, cfg)
        if cfg.get("teacher_mix"):
            # text teacher's probabilities, used only where its top answer matches the gold label
            tf = Path(STAGEB_DIR) / "teacher" / "soft_labels.jsonl"
            soft = {r["id"]: r["probs"] for r in common.read_jsonl(tf) if r.get("teacher_acc_gold") == 1} \
                if tf.exists() else {}
            m, n = cfg["teacher_mix"], 0
            for it in items:
                p = soft.get(it["id"])
                if p is not None and len(p) == len(it["target"]) and abs(sum(p) - 1) < 1e-2:
                    it["target"] = [(1 - m) * a + m * b for a, b in zip(it["target"], p)]
                    n += 1
            print(f"[train] teacher soft labels blended into {n} items (mix {m}); file present: {tf.exists()}")
        if cfg.get("mix"):
            before = len(items)
            items = apply_mix(items, MIXES[cfg["mix"]] if isinstance(cfg["mix"], str) else cfg["mix"])
            print(f"[train] mixture {cfg['mix']}: {before} items -> {len(items)} training passes", flush=True)
        from collections import Counter

        print(f"[train] mix: {len(items)} items {dict(Counter(it['source'] for it in items).most_common())}")
    return items


def fit_and_save_temperature(out: Path, system: str, items: list[dict], preds: list[dict]) -> float:
    """One temperature for the model, fitted on the CAL split with every source weighted equally.
    Per-source temperatures are saved as a diagnostic only."""
    import common

    cal = {it["id"]: it for it in items if it["set"] == "CAL"}
    cal_preds = [p for p in preds if p["id"] in cal]
    sources = [cal[p["id"]]["source"] for p in cal_preds]
    counts = {s: sources.count(s) for s in set(sources)}
    weights = [1.0 / counts[s] for s in sources]
    lps, tgts = [p["log_probs"] for p in cal_preds], [cal[p["id"]]["target"] for p in cal_preds]
    temperature = common.fit_temperature(lps, tgts, weights=weights)
    per_source = {s: round(common.fit_temperature([l for l, x in zip(lps, sources) if x == s],
                                                  [t for t, x in zip(tgts, sources) if x == s]), 4)
                  for s in sorted(counts)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"temperature_{system}.json").write_text(json.dumps(
        {"temperature": temperature, "n_cal": len(cal_preds), "n_by_source": counts, "per_source": per_source}))
    return temperature


def select_eval_items(items: list[dict], run: str) -> list[dict]:
    """Smoke mode keeps the first SMOKE_EVAL_ITEMS of every set (sets are stored in a seeded random order)."""
    if not is_smoke(run):
        return items
    seen, out = {}, []
    for it in items:
        seen[it["set"]] = seen.get(it["set"], 0) + 1
        if seen[it["set"]] <= SMOKE_EVAL_ITEMS:
            out.append(it)
    return out


def estimate_tokens(item: dict, tokenizer, banks: dict) -> int:
    """Prompt length in tokens: text tokens + one token per 40 ms (640 samples) of audio + template."""
    import common

    n = len(tokenizer(common.build_prompt_text(item), add_special_tokens=False)["input_ids"]) + 20
    if item.get("audio") is not None:
        n += -(-banks[item["audio"]["bank"]].index[item["audio"]["key"]][1] // 640) + 2
    return n


def make_batches(order: list[int], lengths: list[int], budget: int, max_batch: int) -> list[list[int]]:
    """Greedy batching of (length-sorted) items so that batch_size * longest_item <= budget tokens."""
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (new_max * (len(cur) + 1) > budget or len(cur) >= max_batch):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def audit(run: str) -> dict:
    """CPU dry run of a run's training and CAL data: counts per source, and checks that no excluded item is left."""
    from collections import Counter

    builders = freeze_builders(run)
    cfg = run_config(run)
    items = load_train_items(run)
    cal = drop_excluded(stageb_items("cal", run), cfg, "cal")

    def bad(it):
        m = it.get("meta") or {}
        where = " ".join(str(x) for x in ((it.get("audio") or {}).get("key"), (it.get("audio") or {}).get("bank"),
                                          m.get("speaker"), m.get("clip"), m.get("corpus"))).lower()
        share_alike = cfg.get("drop_share_alike") and is_share_alike(it, cfg.get("drop_share_alike_backgrounds", False))
        return ("vctk" in where or m.get("corpus") in cfg.get("drop_corpora", ()) or share_alike
                or it.get("source") in cfg.get("drop_sources", ()))

    n_bad, n_bad_cal = sum(map(bad, items)), sum(map(bad, cal))
    corp = Counter((it.get("meta") or {}).get("corpus") for it in items)
    return {"run": run, "builders": builders, "train_passes": len(items),
            "distinct_items": len({it["id"].split("~r")[0] for it in items}),
            "audio_passes": sum(it.get("audio") is not None for it in items),
            "by_source": dict(Counter(it["source"] for it in items).most_common()),
            "by_tagged_corpus": {str(k): v for k, v in corp.most_common()}, "cal_items": len(cal),
            "excluded_left_in_train": n_bad, "excluded_left_in_cal": n_bad_cal,
            "est_steps": len(items) // cfg.get("batch", EFFECTIVE_BATCH)}
