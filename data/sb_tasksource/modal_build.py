"""Build sb_tasksource on Modal (CPU only); ::smoke writes to /vol/stageb_smoke/, which training never reads.

    modal run --detach data/sb_tasksource/modal_build.py::main
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
volume = modal.Volume.from_name("homi", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.3.3", "pyarrow==21.0.0", "pandas==2.3.3", "huggingface_hub==0.35.3", "pyyaml==6.0.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "build.py", "/root/build.py")
    .add_local_file(HERE / "task_licenses.json", "/root/task_licenses.json")
    .add_local_file(HERE / "jev_source_scores.csv", "/root/jev_source_scores.csv")
)
app = modal.App("sb-tasksource-build")

CPU, MEM_MB = 8, 49152
SMOKE = {"files": 2, "train_target": 3000, "cal_target": 100}
ITEM_KEYS = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
COMMON_FREQ_MAX = 25  # a state line found in more groups than this is boilerplate, not a CAL fingerprint


def out_dir(smoke: bool) -> Path:
    return Path(f"{VOL}/stageb_smoke/sb_tasksource") if smoke else Path(f"{VOL}/stageb/sb_tasksource")


def _ref_items() -> dict:
    """Evaluation items the training data must not overlap with."""
    import common as C

    refs = {}
    for name, path in (("adi", f"{VOL}/adi/items.jsonl"), ("exp0_eval", f"{VOL}/data/eval_items.jsonl")):
        refs[name] = C.read_jsonl(path) if Path(path).exists() else None
    return refs


def _strings(x, out: list, min_len: int = 20) -> None:
    if isinstance(x, str):
        if len(x) >= min_len:
            out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _strings(v, out, min_len)
    elif isinstance(x, list):
        for v in x:
            _strings(v, out, min_len)


def leakage_index(refs: dict) -> dict:
    """Content keys of the reference (eval) items: text lines of state_text / meta strings, and option sets that are
    item-specific (an option set shared by > 3 items is a label vocabulary such as emotions, not content)."""
    from collections import Counter

    import build as B

    idx = {}
    for name, items in refs.items():
        if items is None:
            idx[name] = None
            continue
        keys, optsets, questions = set(), Counter(), set()
        for it in items:
            texts = []
            _strings(it.get("state_text"), texts)
            _strings(it.get("meta"), texts)
            for t in texts:
                keys |= {k for k in B.line_keys(t, min_len=20)}
            if len(it.get("options") or []) >= 3:
                optsets[frozenset(B.normalise(o) for o in it["options"])] += 1
            questions.add(B.normalise(it.get("question", "")))
        idx[name] = {"keys": keys, "optsets": {s for s, n in optsets.items() if n <= 3}, "questions": questions,
                     "n_items": len(items)}
    return idx


def leakage_hits(item: dict, idx: dict) -> list[str]:
    import build as B

    hits = []
    item_keys = B.line_keys(item.get("state_text") or "", min_len=20) | {B.normalise(o) for o in item["options"]
                                                                        if len(o) >= 20}
    oset = frozenset(B.normalise(o) for o in item["options"])
    for name, ix in idx.items():
        if ix is None:
            continue
        if item_keys & ix["keys"]:
            hits.append(f"{name}:text")
        if len(item["options"]) >= 3 and oset in ix["optsets"]:
            hits.append(f"{name}:option_set")
    return hits


def validate_items(train: list[dict], cal: list[dict]) -> dict:
    """Schema + label checks on items read back from disk (same rules as training/recipe.valid_items, plus ours)."""
    from collections import Counter

    import numpy as np

    problems, stats = Counter(), Counter()
    ids = Counter()
    for it in train + cal:
        ids[it["id"]] += 1
        if set(it) != ITEM_KEYS:
            problems["schema_keys"] += 1
        k = len(it["options"])
        if not 2 <= k <= 40:
            problems["n_options"] += 1
        if len(it["target"]) != k or abs(sum(it["target"]) - 1) >= 1e-3 or min(it["target"]) < 0:
            problems["target"] += 1
        if not 0 <= it["gold"] < k:
            problems["gold_range"] += 1
        if not isinstance(it["question"], str) or not all(isinstance(o, str) and o for o in it["options"]):
            problems["strings"] += 1
        if it["audio"] is not None or it["channel"] != "clean":
            problems["audio"] += 1
        q = it["meta"].get("qtype")
        if q not in ("choice", "noul", "score"):
            problems["qtype"] += 1
        if q == "noul" and sorted(it["options"]) != ["no", "yes"]:
            problems["noul_options"] += 1
        if not it["meta"]["soft"] and it["gold"] != int(np.argmax(it["target"])):
            problems["gold_not_argmax"] += 1
        if it["meta"]["soft"]:
            stats["soft"] += 1
        if not it["source"].startswith("ts:"):
            problems["source_prefix"] += 1
    problems["duplicate_ids"] = sum(n - 1 for n in ids.values() if n > 1)
    tr_rows = {it["meta"]["row_id"] for it in train}
    tr_groups = {it["meta"]["group_id"] for it in train}
    tr_ex = {it["meta"]["example_id"] for it in train if it["meta"]["example_id"]}
    stats["cal_rows_in_train"] = sum(it["meta"]["row_id"] in tr_rows for it in cal)
    stats["cal_groups_in_train"] = sum(it["meta"]["group_id"] in tr_groups for it in cal)
    stats["cal_examples_in_train"] = sum(it["meta"]["example_id"] in tr_ex for it in cal)
    stats["distinct_sources"] = len({it["source"] for it in train + cal})
    for k in ("cal_rows_in_train", "cal_groups_in_train", "cal_examples_in_train"):
        if stats[k]:
            problems[k] = stats[k]
    if stats["distinct_sources"] >= 60:
        problems["too_many_sources"] = stats["distinct_sources"]
    return {"problems": {k: v for k, v in problems.items() if v}, "stats": dict(stats), "n_checked": len(train) + len(cal)}


@app.function(image=image, volumes={VOL: volume}, secrets=[hf_secret], cpu=CPU, memory=MEM_MB, timeout=3 * 3600)
def build(smoke: bool = False) -> str:
    import csv
    import os
    import random
    from collections import Counter, defaultdict

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import yaml
    from huggingface_hub import HfApi, hf_hub_download

    import build as B
    import common as C

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    OUT = out_dir(smoke)
    train_target = SMOKE["train_target"] if smoke else B.TRAIN_TARGET
    cal_target = SMOKE["cal_target"] if smoke else B.CAL_TARGET

    # Metadata and task-level decisions
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(B.REPO, repo_type="dataset", revision=B.REVISION)
                   if f.startswith(f"{B.CONFIG}/train-") and f.endswith(".parquet"))
    skipped_files = [f for f in files if "webinstruct" in f]  # webinstruct is excluded (see CLOSED_MODEL_TASKS)
    files = [f for f in files if f not in skipped_files][: SMOKE["files"] if smoke else None]
    log(f"{len(files)} parquet files ({B.REPO}@{B.REVISION}, config {B.CONFIG}); skipped {skipped_files}")
    sources_yaml = yaml.safe_load(open(hf_hub_download(B.REPO, "sources.yaml", repo_type="dataset",
                                                       revision=B.REVISION, cache_dir="/tmp/hf")))
    ys = sources_yaml["sources"]
    ts_lic = json.load(open("/root/task_licenses.json"))
    assert ts_lic["tasksource_commit"] == B.TASKSOURCE_COMMIT
    ts_tasks = ts_lic["tasks"]
    scores = {r["source"]: r for r in csv.DictReader(open("/root/jev_source_scores.csv"))}

    def repos_of(y):
        return sorted({r for r in [y.get("dataset"), *(y.get("revisions") or {}), *(y.get("originals") or []),
                                   *(y.get("data_files_from") or [])] if r})

    decisions = {}
    for t in sorted(ys):
        y = ys[t]
        repos = repos_of(y)
        reason = B.leaderboard_match(t, repos) or B.content_overlap_match(t) or B.closed_model_match(t, repos)
        if reason is None and t in scores and float(scores[t]["trust"]) == 0.0:
            reason = "failed tasksource label review (jev_source_scores.csv trust = 0)"
        ok, lreason, rec = B.licence_decision(t, ts_tasks.get(t), y, {y.get("license_use")} - {None})
        if reason is None and not ok:
            reason = lreason
        decisions[t] = {"keep": reason is None, "reason": reason or "ok", "licence": rec, "repos": repos,
                        "revisions": y.get("revisions"), "train_rows_in_build": (y.get("rows") or {}).get("train")}
    keep = sorted(t for t, d in decisions.items() if d["keep"])
    log(f"tasks in build: {len(decisions)}; pass task-level filters: {len(keep)}")

    # Read the train rows of kept tasks
    cols = ["state", "kind", "id", "options", "target", "question", "source", "variant", "split", "group_id",
            "question_id", "example_id", "license_use"]
    tables, row_uses, seen_sources, split_counts = [], defaultdict(Counter), Counter(), Counter()
    keep_arr = pa.array(keep)
    for f in files:
        p = hf_hub_download(B.REPO, f, repo_type="dataset", revision=B.REVISION, cache_dir="/tmp/hf")
        t = pq.read_table(p, columns=cols)
        sl = t.select(["source", "license_use", "split"]).to_pandas()
        seen_sources.update(sl["source"].tolist())
        split_counts.update(sl["split"].tolist())
        for (s, u), n in sl.groupby(["source", "license_use"]).size().items():
            row_uses[s][u] += int(n)
        t = t.filter(pc.and_(pc.is_in(t["source"], value_set=keep_arr), pc.equal(t["split"], "train")))
        tables.append(t)
        os.remove(os.path.realpath(p))
        log(f"read {f}: kept {t.num_rows} rows")
    df = pa.concat_tables(tables).to_pandas()
    del tables
    log(f"candidate rows: {len(df)}; source split values seen: {dict(split_counts)}")
    for s in seen_sources:
        if s not in decisions:
            decisions[s] = {"keep": False, "reason": "not listed in sources.yaml", "licence": {}, "repos": []}
    for t in keep:  # every row of a kept task must say commercial too
        if set(row_uses.get(t, {})) - {"commercial"}:
            decisions[t].update(keep=False, reason=f"row license_use {dict(row_uses[t])}")
        decisions[t]["licence"]["row_license_use"] = dict(row_uses.get(t, {}))
    keep = [t for t in keep if decisions[t]["keep"]]

    # Valid rows per task
    valid, problems = {}, defaultdict(Counter)
    for t, g in df.groupby("source", sort=True):
        if t not in keep:
            continue
        rows = []
        for r in g.to_dict("records"):
            r["options"] = [] if r["options"] is None else list(r["options"])
            r["target"] = [float(x) for x in r["target"]]
            why = B.row_problem(r)
            if why:
                problems[t][why] += 1
            else:
                rows.append(r)
        valid[t] = rows
    del df
    for t in keep:
        if not valid.get(t):
            decisions[t].update(keep=False, reason="no valid train rows after filters (length / options)")
    keep = [t for t in keep if decisions[t]["keep"]]
    avail = {t: len(valid[t]) for t in keep}
    log(f"kept tasks: {len(keep)}; valid rows: {sum(avail.values())}")

    # Quotas, CAL, train
    quota = B.task_quotas(avail, target=train_target)
    tot_q = sum(quota.values())
    cal_q = {t: max(1, round(cal_target * quota.get(t, 0) / tot_q)) for t in keep if quota.get(t, 0) > 0}
    order, pool = {}, {}
    for t in keep:
        gids, groups = B.order_groups(valid[t], seed=f"{C.SEED}:{B.BUILDER}:{t}")
        order[t] = (gids, groups)
        need, n, pl = 3 * (quota.get(t, 0) + cal_q.get(t, 0)) + 50, 0, []
        for g in gids:
            if n >= need:
                break
            pl.append(g)
            n += len(groups[g])
        pool[t] = pl
    key_cache = {}

    def keys_of(r):
        k = key_cache.get(r["id"])
        if k is None:
            k = key_cache[r["id"]] = {hash(x) for x in B.line_keys(r["state"])}
        return k

    freq = Counter()
    for t in keep:
        gids, groups = order[t]
        for g in pool[t]:
            freq.update(set().union(*(keys_of(r) for r in groups[g])))
    log(f"line-key frequency index: {len(freq)} keys")
    cal_rows, cal_gids = {}, {}
    for t in keep:
        if t in cal_q:
            gids, groups = order[t]
            cal_rows[t], cal_gids[t] = B.choose_cal(pool[t], groups, cal_q[t])
    blocked_ex = {r["example_id"] for rows in cal_rows.values() for r in rows if r.get("example_id")}
    blocked_keys = {k for rows in cal_rows.values() for r in rows for k in keys_of(r) if freq[k] <= COMMON_FREQ_MAX}
    train_rows, fill_stats = {}, Counter()
    for t in keep:
        if quota.get(t, 0) <= 0:
            continue
        gids, groups = order[t]
        train_rows[t], st = B.fill_train(gids, groups, quota[t], cal_gids.get(t, set()), blocked_ex, blocked_keys,
                                         keys_of=keys_of)
        fill_stats.update(st)
    log(f"sampled train rows {sum(map(len, train_rows.values()))}, CAL rows {sum(map(len, cal_rows.values()))}; "
        f"{dict(fill_stats)}")

    # Convert
    def meta_extra(t):
        lic = decisions[t]["licence"]
        return {"task": t, "family": B.family_of(t), "bucket": B.bucket_of(t), "licence": ", ".join(lic["licences"]),
                "licence_use": "commercial", "share_alike": lic["share_alike"],
                "task_revisions": decisions[t].get("revisions")}

    train = [B.convert_row(r, set_name="train", source="ts:?", meta_extra=meta_extra(t))
             for t in sorted(train_rows) for r in train_rows[t]]
    cal = [B.convert_row(r, set_name="CAL", source="ts:?", meta_extra=meta_extra(t))
           for t in sorted(cal_rows) for r in cal_rows[t]]
    train, n_noul_dropped = B.rebalance_noul(train, seed=f"{C.SEED}:{B.BUILDER}:noul")
    cal, n_noul_dropped_cal = B.rebalance_noul(cal, seed=f"{C.SEED}:{B.BUILDER}:noul-cal", per_task=False)
    log(f"converted: train {len(train)} (noul rebalance dropped {n_noul_dropped}), CAL {len(cal)} "
        f"(dropped {n_noul_dropped_cal})")

    # Leakage against the evaluation items
    refs = _ref_items()
    idx = leakage_index(refs)
    leak = Counter()
    kept_train, kept_cal, leak_examples = [], [], []
    for lst, out in ((train, kept_train), (cal, kept_cal)):
        for it in lst:
            hits = leakage_hits(it, idx)
            if hits:
                leak.update(hits)
                if len(leak_examples) < 10:
                    leak_examples.append({"id": it["id"], "task": it["meta"]["task"], "hits": hits,
                                          "options": it["options"][:6], "state": (it["state_text"] or "")[:200]})
            else:
                out.append(it)
    train, cal = kept_train, kept_cal
    q_overlap = Counter()
    for it in train + cal:
        for name, ix in idx.items():
            if ix is not None and B.normalise(it["question"]) in ix["questions"]:
                q_overlap[name] += 1
    leakage = {"reference_sets": {k: (None if v is None else v["n_items"]) for k, v in idx.items()},
               "dropped_items": dict(leak), "examples": leak_examples,
               "template_question_text_equal_to_a_reference_question (kept: generic wording, no shared content)":
                   dict(q_overlap),
               "audio_ids": "not applicable: text-only items (no audio, speakers or recordings)"}
    log(f"leakage: {leakage['dropped_items']}; question-text equality {dict(q_overlap)}")

    # Sources, ids, write
    fam2src = B.assign_sources(train + cal)
    for it in train + cal:
        it["source"] = fam2src[it["meta"]["family"]]
    seen = Counter()
    for it in train + cal:
        seen[it["id"]] += 1
        if seen[it["id"]] > 1:
            it["id"] = f"{it['id']}#{seen[it['id']]}"
    random.Random(f"{C.SEED}:{B.BUILDER}:order").shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    if (OUT / "summary.json").exists():
        (OUT / "summary.json").unlink()  # summary.json is written last and marks a finished build
    C.write_jsonl(OUT / "items_train.jsonl", train)
    C.write_jsonl(OUT / "items_cal.jsonl", cal)
    volume.commit()
    rb_train, rb_cal = C.read_jsonl(OUT / "items_train.jsonl"), C.read_jsonl(OUT / "items_cal.jsonl")
    validation = validate_items(rb_train, rb_cal)
    # Re-check on the written files that no CAL text line appears in train
    cal_keys = {k for it in rb_cal for k in {hash(x) for x in B.line_keys(it["state_text"] or "")}
                if freq.get(k, 0) <= COMMON_FREQ_MAX}
    validation["stats"]["train_items_sharing_a_cal_text_line"] = sum(
        bool({hash(x) for x in B.line_keys(it["state_text"] or "")} & cal_keys) for it in rb_train)
    log(f"validation: {validation}")

    # Summary (written last)
    def counts(items, key):
        return dict(Counter(key(it) for it in items).most_common())

    noul_bal = {}
    for name, items in (("train", rb_train), ("CAL", rb_cal)):
        hard = [it for it in items if it["meta"]["qtype"] == "noul" and not it["meta"]["soft"]]
        noul_bal[name] = {"hard_noul": len(hard),
                          "yes_share": round(sum(it["options"][it["gold"]] == "yes" for it in hard) / max(1, len(hard)), 4),
                          "soft_noul": sum(it["meta"]["qtype"] == "noul" and it["meta"]["soft"] for it in items)}
    per_task = {}
    for t in keep:
        per_task[t] = {"train": 0, "cal": 0, "family": B.family_of(t), "bucket": B.bucket_of(t),
                       "source": fam2src.get(B.family_of(t)), "licences": decisions[t]["licence"]["licences"],
                       "share_alike": decisions[t]["licence"]["share_alike"], "revisions": decisions[t].get("revisions"),
                       "valid_rows": avail[t], "row_skips": dict(problems[t])}
    for it in rb_train:
        per_task[it["meta"]["task"]]["train"] += 1
    for it in rb_cal:
        per_task[it["meta"]["task"]]["cal"] += 1
    lic_tasks = Counter(l for t in keep for l in decisions[t]["licence"]["licences"])
    examples = {}
    for it in rb_train:
        ex = examples.setdefault(it["source"], [])
        if len(ex) < 3:
            ex.append({"task": it["meta"]["task"], "qtype": it["meta"]["qtype"], "question": it["question"],
                       "options": it["options"], "gold": it["gold"], "target": it["target"],
                       "state": (it["state_text"] or "")[:300]})
    elapsed = time.time() - t0
    summary = {
        "builder": B.BUILDER, "smoke": smoke, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"repo": B.REPO, "revision": B.REVISION, "config": B.CONFIG, "split_used": "train (source split == 'train')",
                    "files": files, "skipped_files": skipped_files,
                    "licence_note": "card license: other; each source keeps its own licence (filtered below)"},
        "tasksource_package": {"repo": "github.com/sileod/tasksource", "commit": B.TASKSOURCE_COMMIT,
                               "licence_api": "task_licenses() snapshot in data/sb_tasksource/task_licenses.json",
                               "code_licence": "tasksource code; bucket regexes copied from jev_mixes.py with attribution"},
        "counts": {"train": len(rb_train), "cal": len(rb_cal), "tasks_kept": len(keep),
                   "tasks_in_build": len(decisions), "audio_hours": 0.0},
        "by_source": {"train": counts(rb_train, lambda it: it["source"]), "cal": counts(rb_cal, lambda it: it["source"])},
        "by_qtype": {"train": counts(rb_train, lambda it: it["meta"]["qtype"]),
                     "cal": counts(rb_cal, lambda it: it["meta"]["qtype"])},
        "by_source_qtype": counts(rb_train + rb_cal, lambda it: f"{it['set']}|{it['source']}|{it['meta']['qtype']}"),
        "by_bucket": counts(rb_train, lambda it: it["meta"]["bucket"]),
        "by_variant": counts(rb_train, lambda it: it["meta"]["variant"]),
        "soft_targets": {"train": sum(it["meta"]["soft"] for it in rb_train), "cal": sum(it["meta"]["soft"] for it in rb_cal)},
        "noul_balance": noul_bal,
        "noul_rebalance_dropped": {"train": n_noul_dropped, "cal": n_noul_dropped_cal},
        "family_to_source": fam2src,
        "licences": {"rule": ("keep a task only if tasksource task_licenses() license_use, sources.yaml license_use and "
                              "every row's license_use are all 'commercial' AND every licence recorded for it (Hub cards "
                              "+ DPI) is CC0/PD, CC BY, CC BY-SA (flagged), Apache-2.0, MIT, BSD, ODC-BY or ODbL; "
                              "'unspecified', 'other', bare 'cc', 'Custom', GPL-family, AFL, OANC etc. are dropped"),
                     "tasks_per_licence": dict(lic_tasks.most_common()),
                     "share_alike_tasks": sorted(t for t in keep if decisions[t]["licence"]["share_alike"])},
        "per_task": per_task,
        "dropped_tasks": {t: {"reason": d["reason"], "licence": d["licence"].get("tasksource_license") or
                              d["licence"].get("card_license"), "repos": d.get("repos")}
                          for t, d in sorted(decisions.items()) if not d["keep"]},
        "dropped_reason_counts": dict(Counter(d["reason"].split(":")[0] for d in decisions.values() if not d["keep"])),
        "row_skip_counts": dict(sum((problems[t] for t in problems), Counter())),
        "cal_construction": {"rule": "whole held-out groups (non-packed rows, <= 2 per group); train never shares a "
                                     "group_id, example_id or a (non-boilerplate) state line with CAL",
                             "train_groups_skipped": dict(fill_stats), "cal_targets": "exact one-hot for hard labels"},
        "label_rules": {"hard": f"one-hot with label smoothing {B.LABEL_SMOOTHING} (train)",
                        "soft": "source distribution kept (votes / mean ratings / probabilities)",
                        "noul": "options ['yes','no'] shuffled by make_item; target [p, 1-p]",
                        "score": "levels kept low -> high (no shuffle)",
                        "choice": "make_item shuffle, except options citing other options by position (order kept)"},
        "limits": {"max_state_chars": B.MAX_STATE_CHARS, "max_total_chars": B.MAX_TOTAL_CHARS,
                   "max_options": B.MAX_OPTIONS, "task_cap": B.TASK_CAP, "family_cap": B.FAMILY_CAP,
                   "multilingual_share": B.MULTILINGUAL_SHARE},
        "leakage_check": leakage,
        "validation": validation,
        "examples": examples,
        "runtime_s": round(elapsed),
        "modal_cost_estimate_usd": round(elapsed / 3600 * (CPU * 0.0473 + MEM_MB / 1024 * 0.008), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    volume.commit()
    log(f"wrote {OUT}")
    return json.dumps({k: summary[k] for k in ("counts", "by_qtype", "noul_balance", "soft_targets", "validation",
                                               "dropped_reason_counts", "modal_cost_estimate_usd")} |
                      {"n_sources": len(summary["by_source"]["train"]), "leakage": leakage["dropped_items"]},
                      indent=1, default=str)


@app.function(image=image, volumes={VOL: volume}, cpu=2, memory=16384, timeout=1800)
def validate_remote(smoke: bool = False) -> str:
    import common as C

    OUT = out_dir(smoke)
    train, cal = C.read_jsonl(OUT / "items_train.jsonl"), C.read_jsonl(OUT / "items_cal.jsonl")
    res = validate_items(train, cal)
    s = json.loads((OUT / "summary.json").read_text())
    res["summary_counts"] = s["counts"]
    return json.dumps(res, indent=1)


@app.local_entrypoint()
def smoke():
    print(build.remote(smoke=True))


@app.local_entrypoint()
def main():
    print(build.remote(smoke=False))


@app.local_entrypoint()
def validate(smoke: bool = False):
    print(validate_remote.remote(smoke=smoke))
