#!/usr/bin/env python3
"""Audit a small SWE-Gym OpenHands baseline pool without publishing patches."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path("/data/chenduo/workspace/agentic_rl/qwen3_swe_rl")
DATASET_FINGERPRINT = "3c18e7f6e3af6f68"
CANDIDATES = [
    "getmoto__moto-6785",
    "getmoto__moto-6953",
    "pydantic__pydantic-5529",
    "pydantic__pydantic-5591",
    "pydantic__pydantic-5744",
    "dask__dask-7418",
    "conan-io__conan-10213",
    "conan-io__conan-11799",
]
SELECTED = [
    "pydantic__pydantic-5529",
    "pydantic__pydantic-5591",
    "getmoto__moto-6785",
]


def parsed_list(value):
    return json.loads(value) if isinstance(value, str) else value


def ids_from_json(path):
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        for key in ("tasks", "instances", "instance_ids"):
            if key in obj:
                obj = obj[key]
                break
    if obj and isinstance(obj[0], dict):
        return {item["instance_id"] for item in obj}
    return set(obj)


def diff_summary(text):
    text = text or ""
    return {
        "files": re.findall(r"^diff --git a/(.*?) b/", text, flags=re.M),
        "hunk_headers": [line for line in text.splitlines() if line.startswith("@@")],
        "added_lines": sum(1 for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")),
        "deleted_lines": sum(1 for line in text.splitlines() if line.startswith("-") and not line.startswith("---")),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


records = {
    row["instance_id"]: row
    for row in json.loads((ROOT / ".private/swe_gym_reference_records.json").read_text())
}
validated_doc = json.loads((ROOT / "data/swegym_validated_tasks.json").read_text())
validated = {row["instance_id"]: row for row in validated_doc["tasks"]}
train_ids = ids_from_json(ROOT / "data/swe_gym_train_100.json")
dev_ids = ids_from_json(ROOT / "data/swe_internal_dev_20.json")
sft_seen_ids = ids_from_json(ROOT / "data/openhands7b_sft_seen_instance_ids.json")

manual_decisions = {
    "dask__dask-7418": {
        "status": "EXCLUDE_SEMANTIC_MISMATCH",
        "reason": "Problem asks for HDFS CI workflow behavior, while gold/test files concern dask.bag.read_text unicode behavior.",
    },
    "conan-io__conan-10213": {
        "status": "EXCLUDE_BASELINE_V2",
        "reason": "Already exhausted by G=1 and G=8; CMake/build-system task is not useful for the next diversity sanity.",
    },
    "conan-io__conan-11799": {
        "status": "EXCLUDE_BASELINE_V2",
        "reason": "CMake/Android NDK build-system task conflicts with the requested lightweight non-build-system pilot.",
    },
    "pydantic__pydantic-5744": {
        "status": "DEFER_COMPLEX_PATCH",
        "reason": "Semantically aligned but has a large multi-file gold/test delta and 9 FAIL_TO_PASS tests.",
    },
    "getmoto__moto-6953": {
        "status": "RESERVE",
        "reason": "Semantically aligned but touches three source files and has 3 FAIL_TO_PASS tests; keep as reserve.",
    },
    "pydantic__pydantic-5529": {
        "status": "SELECT",
        "reason": "Semantically aligned, small source delta, 2 FAIL_TO_PASS and only 3 PASS_TO_PASS tests.",
    },
    "pydantic__pydantic-5591": {
        "status": "SELECT",
        "reason": "Semantically aligned and single-source-file patch with 1 FAIL_TO_PASS test.",
    },
    "getmoto__moto-6785": {
        "status": "SELECT",
        "reason": "Semantically aligned auth failure, modest two-file source delta and 1 FAIL_TO_PASS test.",
    },
}

audited = []
for task_id in CANDIDATES:
    record = records[task_id]
    gate = validated[task_id]
    patch = diff_summary(record.get("patch"))
    test_patch = diff_summary(record.get("test_patch"))
    verification = gate["verification"]
    gate_ok = (
        verification["base_hidden_test_patch"] == "FAIL"
        and verification["gold_patch"] == "PASS"
        and verification["reset_hidden_test_patch"] == "FAIL"
    )
    audited.append({
        "instance_id": task_id,
        "repo": record["repo"],
        "version": record["version"],
        "base_commit": record["base_commit"],
        "problem_title": (record.get("problem_statement") or "").splitlines()[0],
        "problem_statement_sha256": hashlib.sha256((record.get("problem_statement") or "").encode()).hexdigest(),
        "source_patch": patch,
        "test_patch": test_patch,
        "fail_to_pass": parsed_list(record["FAIL_TO_PASS"]),
        "pass_to_pass_count": len(parsed_list(record["PASS_TO_PASS"])),
        "verifier_gate": "PASS" if gate_ok else "FAIL",
        "verifier_result_file": verification["result_file"],
        "image_digest": gate["image_digest"],
        **manual_decisions[task_id],
    })

report = {
    "schema_version": 1,
    "source": "SWE-Gym/SWE-Gym",
    "split": "train",
    "dataset_fingerprint": DATASET_FINGERPRINT,
    "selection_purpose": "OpenHands-7B baseline sanity only; not benchmark reporting or checkpoint selection",
    "selection_uses_gold_metadata": True,
    "selection_bias_notice": "Gold/test metadata is used only to reject corrupt or oversized engineering-pilot tasks. Results must not be reported as an unbiased SWE-Gym score.",
    "agent_visibility": "problem_statement only; patch, test_patch, FAIL_TO_PASS and audit rationale are never injected into the agent trajectory",
    "selected_ids": SELECTED,
    "all_selected_in_train_100": set(SELECTED) <= train_ids,
    "selected_internal_dev_overlap": sorted(set(SELECTED) & dev_ids),
    "selected_openhands7b_sft_seen_overlap": sorted(set(SELECTED) & sft_seen_ids),
    "reserve_ids": ["getmoto__moto-6953"],
    "excluded_ids": ["dask__dask-7418", "conan-io__conan-10213", "conan-io__conan-11799"],
    "deferred_ids": ["pydantic__pydantic-5744"],
    "audited": audited,
}

(ROOT / ".private/swegym_openhands_pool_v2_audit.json").write_text(json.dumps(report, indent=2) + "\n")

public_tasks = []
for task_id in SELECTED:
    item = validated[task_id]
    public_tasks.append({
        "instance_id": task_id,
        "repo": item["repo"],
        "version": item["version"],
        "base_commit": item["base_commit"],
        "eval_image": item["eval_image"],
        "image_digest": item["image_digest"],
        "verifier_gate": "FAIL_GOLD-PASS_RESET-FAIL",
        "network_disabled": True,
        "in_fixed_train_100": task_id in train_ids,
        "in_internal_dev_20": task_id in dev_ids,
        "in_openhands7b_sft_seen_ids": task_id in sft_seen_ids,
    })

manifest = {
    "schema_version": 1,
    "source": "SWE-Gym/SWE-Gym",
    "split": "train",
    "dataset_fingerprint": DATASET_FINGERPRINT,
    "purpose": "OpenHands-7B corrected G=1 baseline sanity",
    "count": len(public_tasks),
    "tasks": public_tasks,
}
(ROOT / "data/swegym_openhands_baseline_pool_v2.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"selected_ids": SELECTED, "audit_count": len(audited)}, indent=2))
