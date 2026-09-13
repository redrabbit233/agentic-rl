#!/usr/bin/env python3
"""Create deterministic, non-overlapping SWE-Gym train/dev manifests.

Public manifests deliberately omit both gold and test patches.  The verifier's
private reference records are Git-ignored and must never be supplied to an
agent prompt.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET = "SWE-Gym/SWE-Gym"
SPLIT = "train"
SEED = "qwen3-swe-rl-swegym-v1"
REPOS = (
    "getmoto/moto",
    "pydantic/pydantic",
    "conan-io/conan",
    "dask/dask",
)
PER_REPO = 30
TRAIN_PER_REPO = 25
ROOT = Path(__file__).resolve().parents[1]


def rank(instance_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{instance_id}".encode()).hexdigest()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_tests(value: str | list[str]) -> list[str]:
    return json.loads(value) if isinstance(value, str) else value


def public_record(record: dict) -> dict:
    return {
        "instance_id": record["instance_id"],
        "repo": record["repo"],
        "base_commit": record["base_commit"],
        "version": record["version"],
        "problem_statement": record["problem_statement"],
        "problem_statement_sha256": digest(record["problem_statement"]),
        "fail_to_pass_count": len(parse_tests(record["FAIL_TO_PASS"])),
        "pass_to_pass_count": len(parse_tests(record["PASS_TO_PASS"])),
        "provenance": {"dataset": DATASET, "split": SPLIT, "seed": SEED},
    }


def private_record(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "instance_id", "repo", "base_commit", "version", "patch",
            "test_patch", "problem_statement", "hints_text", "created_at",
            "FAIL_TO_PASS", "PASS_TO_PASS",
        )
    }


def valid(record: dict, missing_images: set[str]) -> bool:
    return (
        record["instance_id"] not in missing_images
        and record["repo"] in REPOS
        and all(record.get(key) for key in ("problem_statement", "base_commit", "patch", "test_patch", "version"))
        and bool(parse_tests(record["FAIL_TO_PASS"]))
    )


def main() -> None:
    missing_path = ROOT / "third_party" / "env-swe-gym" / "missing_images.txt"
    missing_images = set(missing_path.read_text().split())
    dataset = load_dataset(DATASET, split=SPLIT)
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    by_repo: dict[str, list[dict]] = {repo: [] for repo in REPOS}
    for record in dataset:
        if valid(record, missing_images):
            by_repo[record["repo"]].append(record)

    selected: list[dict] = []
    for repo in REPOS:
        choices = sorted(by_repo[repo], key=lambda r: rank(r["instance_id"]))
        if len(choices) < PER_REPO:
            raise RuntimeError(f"{repo}: only {len(choices)} eligible tasks")
        selected.extend(choices[:PER_REPO])

    train, dev = [], []
    for repo in REPOS:
        repo_records = [r for r in selected if r["repo"] == repo]
        train.extend(repo_records[:TRAIN_PER_REPO])
        dev.extend(repo_records[TRAIN_PER_REPO:])
    if len(train) != 100 or len(dev) != 20:
        raise RuntimeError(f"Unexpected split sizes: train={len(train)}, dev={len(dev)}")
    if {r["instance_id"] for r in train} & {r["instance_id"] for r in dev}:
        raise RuntimeError("train/dev overlap")

    data_dir, private_dir = ROOT / "data", ROOT / ".private"
    data_dir.mkdir(exist_ok=True)
    private_dir.mkdir(exist_ok=True)
    (data_dir / "swe_gym_train_100.json").write_text(json.dumps([public_record(r) for r in train], indent=2) + "\n")
    (data_dir / "swe_internal_dev_20.json").write_text(json.dumps([public_record(r) for r in dev], indent=2) + "\n")
    # Exercise image/verification compatibility across all four repositories
    # before scaling.  The train/dev split itself is unchanged.
    smoke = []
    for index, repo in enumerate(REPOS):
        quota = 3 if index < 2 else 2
        smoke.extend([r for r in train if r["repo"] == repo][:quota])
    (data_dir / "swe_train_smoke_10.json").write_text(json.dumps([public_record(r) for r in smoke], indent=2) + "\n")
    (private_dir / "swe_gym_reference_records.json").write_text(
        json.dumps([private_record(r) for r in train + dev], indent=2) + "\n"
    )
    manifest = {
        "dataset": DATASET,
        "split": SPLIT,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": SEED,
        "repos": list(REPOS),
        "excluded_missing_images": len(missing_images),
        "train_count": len(train),
        "internal_dev_count": len(dev),
        "smoke_count": len(smoke),
        "repo_counts": Counter(r["repo"] for r in train + dev),
        "private_reference_records": ".private/swe_gym_reference_records.json",
        "private_reference_records_gitignored": True,
    }
    (data_dir / "swe_gym_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2, default=dict))


if __name__ == "__main__":
    main()
