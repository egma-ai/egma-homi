"""Snapshot tasksource's per-task licences (pinned commit) into task_licenses.json. Runs locally, no data download:

    uv run --python 3.12 --with "tasksource @ git+https://github.com/sileod/tasksource@d4f939cdaa900a7b70c6d4dd9f4f77a10a6306b7" \
        --with pillow python data/sb_tasksource/licences_snapshot.py
"""
from __future__ import annotations

import json
from pathlib import Path

TASKSOURCE_COMMIT = "d4f939cdaa900a7b70c6d4dd9f4f77a10a6306b7"


def main() -> None:
    from tasksource import list_tasks, task_licenses

    out = {"tasksource_commit": TASKSOURCE_COMMIT, "tasks": {}, "list_tasks_commercial": []}
    for multilingual in (False, True):
        df = task_licenses(multilingual=multilingual)
        for r in df.to_dict("records"):
            key = ("multilingual/" if multilingual else "") + r["id"]
            out["tasks"][key] = {"license": r["license"], "license_use": r["license_use"],
                                 "card_licenses": r.get("card_licenses") or {}, "dpi_licenses": list(r.get("dpi_licenses") or [])}
        lt = list_tasks(license_use="commercial", multilingual=multilingual)
        out["list_tasks_commercial"] += [("multilingual/" if multilingual else "") + i for i in lt["id"]]
    path = Path(__file__).with_name("task_licenses.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(f"wrote {path}: {len(out['tasks'])} tasks, {len(out['list_tasks_commercial'])} commercial via list_tasks")


if __name__ == "__main__":
    main()
