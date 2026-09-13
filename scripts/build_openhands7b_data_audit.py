#!/usr/bin/env python3
"""Map public OpenHands SFT trajectories to SWE-Gym IDs without heuristics."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SFT_REPO = "SWE-Gym/OpenHands-SFT-Trajectories"
SFT_REVISION = "4aaa5a4a4b5861f4799d2336908760c190ac3b17"
SFT_SPLIT = "train.success.oss"
GYM_REPO = "SWE-Gym/SWE-Gym"
GYM_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def extract_statement(messages: list[dict]) -> str:
    for message in messages:
        match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", message.get("content", ""), re.S)
        if match:
            return match.group(1)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    gym = load_dataset(GYM_REPO, revision=GYM_REVISION, split="train", streaming=True, cache_dir=str(args.cache_dir))
    statement_to_ids: dict[str, list[str]] = defaultdict(list)
    gym_count = 0
    for row in gym:
        statement_to_ids[normalize(row["problem_statement"])].append(row["instance_id"])
        gym_count += 1
    sft = load_dataset(SFT_REPO, revision=SFT_REVISION, split=SFT_SPLIT, streaming=True, cache_dir=str(args.cache_dir))
    exact_matched: set[str] = set()
    conservatively_seen: set[str] = set()
    unmatched = 0
    ambiguous = 0
    trajectories = 0
    for row in sft:
        trajectories += 1
        ids = statement_to_ids.get(normalize(extract_statement(row["messages"])), [])
        if len(ids) == 1:
            exact_matched.add(ids[0])
            conservatively_seen.add(ids[0])
        elif len(ids) > 1:
            ambiguous += 1
            conservatively_seen.update(ids)
        else:
            unmatched += 1
    audit = {
        "sft_repo": SFT_REPO,
        "sft_revision": SFT_REVISION,
        "sft_split": SFT_SPLIT,
        "swe_gym_repo": GYM_REPO,
        "swe_gym_revision": GYM_REVISION,
        "swe_gym_rows": gym_count,
        "sft_trajectory_count": trajectories,
        "exactly_mapped_trajectory_count": trajectories - unmatched - ambiguous,
        "unmatched_trajectory_count": unmatched,
        "ambiguous_trajectory_count": ambiguous,
        "exactly_mapped_unique_instance_count": len(exact_matched),
        "sft_seen_unique_instance_count": len(conservatively_seen),
        "mapping_method": "exact normalized equality of public <pr_description> and SWE-Gym problem_statement; ambiguous text matches conservatively exclude every candidate ID; no repo/version heuristic",
    }
    data = args.project_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "openhands7b_sft_seen_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (data / "openhands7b_sft_seen_instance_ids.json").write_text(json.dumps({"instance_ids": sorted(conservatively_seen)}, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
