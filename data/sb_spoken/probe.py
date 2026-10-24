"""One-off look at the JADE-Data and SLUE-HVB rows (not part of the build): modal run data/sb_spoken/probe.py"""
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("numpy==2.3.3", "pyarrow==21.0.0", "huggingface_hub==0.35.3"))
volume = modal.Volume.from_name("egma-homi")
app = modal.App("sb-spoken-probe")


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, memory=16384,
              secrets=[modal.Secret.from_name("huggingface")])
def peek():
    import collections, io, json, urllib.request
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    # sb_jade item example
    with open("/vol/stageb/sb_jade/items_train.jsonl") as f:
        for i, ln in enumerate(f):
            if i < 2:
                print("SBJADE", ln[:600])
            else:
                break
    p = hf_hub_download("theunnecessarythings/JADE-Data", "train.jsonl", repo_type="dataset",
                        revision="ca75bcb94afd60c4e9c0c17133da4c731e527ea2", cache_dir="/tmp/hf")
    by = collections.defaultdict(list)
    n = 0
    with open(p) as f:
        for ln in f:
            r = json.loads(ln)
            n += 1
            by[r["suite"]].append(r)
    print("JADE rows", n)
    for s, rows in sorted(by.items()):
        qt = collections.Counter(r["question"]["type"] for r in rows)
        sl = sorted(len(json.dumps(r["state"])) for r in rows)
        print("SUITE", s, len(rows), dict(qt), "statelen p50", sl[len(sl)//2], "p10", sl[len(sl)//10])
        for r in rows[:2]:
            q = r["question"]
            crit = q.get("criteria")
            print("   STATE", json.dumps(r["state"])[:400])
            print("   Q", q.get("instructions"), "| type", q["type"], "| ncrit", len(crit) if crit else None,
                  "| crit", json.dumps(crit)[:300], "| target", r["target"], "| keys", list(q.keys()))
    # HVB train
    for i in range(2):
        url = f"https://huggingface.co/api/datasets/asapp/slue-phase-2/parquet/hvb/train/{i}.parquet"
        with urllib.request.urlopen(url, timeout=600) as rr:
            data = rr.read()
        t = pq.read_table(io.BytesIO(data))
        print("HVB part", i, len(data) / 1e6, "MB", t.num_rows, t.schema)
        rows = t.select(["issue_id", "speaker_id", "role", "channel", "intent", "dialog_acts", "duration_ms"]).to_pylist()
        print(" speakers", sorted(collections.Counter(str(r["speaker_id"]) for r in rows).items()))
        print(" calls", len({r["issue_id"] for r in rows}), "channel", collections.Counter(r["channel"] for r in rows))
        a = t.column("audio")[0].as_py()
        print(" audio keys", {k: (type(v), (len(v) if hasattr(v, '__len__') else v)) for k, v in a.items()})


@app.local_entrypoint()
def main():
    peek.remote()
