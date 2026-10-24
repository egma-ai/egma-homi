"""Quick look at tasksource-jev-typed-decisions (pinned revision) on a Modal CPU container.

    modal run data/sb_tasksource/probe.py
"""
from __future__ import annotations

import json

import modal

REPO = "tasksource/tasksource-jev-typed-decisions"
REV = "5d3d8a557f62809d690f22c2c696b44c266d1ada"

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "numpy==2.3.3", "pyarrow==21.0.0", "huggingface_hub==0.35.3", "pandas==2.3.3")
app = modal.App("sb-tasksource-probe")
hf_secret = modal.Secret.from_name("huggingface")


@app.function(image=image, secrets=[hf_secret], cpu=4, memory=16384, timeout=1800)
def probe() -> str:
    from collections import Counter, defaultdict

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO, "full/train-00000-of-00013.parquet", repo_type="dataset", revision=REV,
                        cache_dir="/tmp/hf")
    t = pq.read_table(p)
    out = {"n": t.num_rows, "schema": str(t.schema)}
    df = t.to_pandas()
    out["split"] = Counter(df["split"]).most_common()
    out["kind"] = Counter(df["kind"]).most_common()
    out["variant"] = Counter(df["variant"]).most_common()
    out["license_use"] = Counter(df["license_use"]).most_common()
    out["n_sources"] = df["source"].nunique()
    df["state_len"] = df["state"].str.len()
    df["n_opt"] = df["options"].map(len)
    out["state_len_q"] = df["state_len"].quantile([0.5, 0.9, 0.99]).tolist()
    out["n_opt_by_kind"] = {k: g["n_opt"].describe().to_dict() for k, g in df.groupby("kind")}
    g = df.groupby("group_id").size()
    out["group_size"] = g.describe().to_dict()
    # how hard / soft are targets
    def hard(t):
        return any(abs(x - 1.0) < 1e-9 for x in t) and all(abs(x) < 1e-9 or abs(x - 1.0) < 1e-9 for x in t)
    df["hard"] = df["target"].map(lambda t: hard(list(t)))
    out["hard_by_kind"] = df.groupby("kind")["hard"].mean().to_dict()
    noul = df[df.kind == "noul"]
    out["noul_target_len"] = Counter(noul["target"].map(len)).most_common()
    out["noul_p"] = Counter(noul["target"].map(lambda t: round(float(t[0]), 1))).most_common()
    out["example_id_null"] = int(df["example_id"].isna().sum())
    # does an example_id span several group_ids?
    eg = df.dropna(subset=["example_id"]).groupby("example_id")["group_id"].nunique()
    out["example_in_multi_groups"] = int((eg > 1).sum())
    samples = []
    for (kind, variant), grp in df.groupby(["kind", "variant"]):
        r = grp.iloc[0]
        samples.append({k: (list(r[k]) if k in ("options", "target") else r[k]) for k in
                        ("source", "kind", "variant", "question", "options", "target", "group_id", "question_id",
                         "example_id", "license", "license_use", "split")} | {"state": r["state"][:500]})
    out["samples"] = samples
    # per-source summary for this shard
    out["per_source"] = {s: {"n": len(g2), "kinds": dict(Counter(g2.kind)), "lic": g2.license_use.iloc[0]}
                         for s, g2 in list(df.groupby("source"))[:400]}
    return json.dumps(out, default=str)


@app.local_entrypoint()
def main():
    r = json.loads(probe.remote())
    with open("probe.json", "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in r.items() if k not in ("samples", "per_source")}, indent=1, default=str))
