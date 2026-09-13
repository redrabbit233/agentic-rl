#!/usr/bin/env python3
"""Freeze successful SWE-Gym verifier tasks without exposing patch contents."""
from __future__ import annotations

import json
import re
from pathlib import Path

import docker


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "swegym_validated_tasks.json"
LOGROOT = ROOT / "logs" / "run_evaluation"
records = {
    x["instance_id"]: x
    for x in json.loads((ROOT / ".private" / "swe_gym_reference_records.json").read_text())
}
client = docker.from_env()
validated = []

for result_path in sorted((ROOT / "results").glob("swegym_validation_*.json")):
    result = json.loads(result_path.read_text())
    statuses = [result.get(label, {}).get("resolved") for label in ("base", "gold", "reset")]
    if statuses != [False, True, False]:
        continue
    task_id = result["instance_id"]
    record = records[task_id]
    image = client.images.get(f"sweb.eval.x86_64.{task_id}:latest")
    runtimes = {}
    for label in ("base", "gold", "reset"):
        log = LOGROOT / f"swegym_validation_{label}" / f"swegym_{label}" / task_id / "run_instance.log"
        match = re.search(r"Test runtime: ([0-9.]+) seconds", log.read_text()) if log.exists() else None
        runtimes[label] = float(match.group(1)) if match else None
    validated.append({
        "instance_id": task_id,
        "repo": record["repo"],
        "base_commit": record["base_commit"],
        "version": record["version"],
        "eval_image": result["image"],
        "local_image_tag": f"sweb.eval.x86_64.{task_id}:latest",
        "image_digest": (image.attrs.get("RepoDigests") or [None])[0],
        "verification": {
            "base_hidden_test_patch": "FAIL",
            "gold_patch": "PASS",
            "reset_hidden_test_patch": "FAIL",
            "result_file": str(result_path.relative_to(ROOT)),
            "runtime_seconds": runtimes,
        },
        "network_disabled": True,
        "patches_excluded_from_agent_data": True,
    })

OUT.write_text(json.dumps({
    "source": "SWE-Gym/SWE-Gym",
    "verifier": "SWE-Bench-Fork eval script on prebuilt SWE-Gym eval image",
    "validated_count": len(validated),
    "tasks": validated,
}, indent=2) + "\n")
print(f"wrote {OUT} ({len(validated)} tasks)")
