#!/usr/bin/env python3
"""Validate one SWE-Gym task with the official SWE-Bench-Fork verifier."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import docker
import swebench.harness.run_validation as validation
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec import TestSpec, make_eval_script_list


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = sys.argv[1]
records = json.loads((ROOT / ".private/swe_gym_reference_records.json").read_text())
record = next(x for x in records if x["instance_id"] == TASK_ID)


def make_eval_only_spec(instance: dict) -> TestSpec:
    """Make only the verifier script; SWE-Gym images already contain setup."""
    repo = instance["repo"].lower()
    version = instance["version"]
    base_commit = instance["base_commit"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    parse_list = lambda key: json.loads(instance[key]) if isinstance(instance[key], str) else instance[key]
    return TestSpec(
        instance_id=instance["instance_id"].lower(),
        repo=repo,
        version=version,
        repo_script_list=[],
        env_script_list=[],
        eval_script_list=make_eval_script_list(
            instance, specs, "testbed", "/testbed", base_commit, instance["test_patch"]
        ),
        arch="x86_64",
        FAIL_TO_PASS=parse_list("FAIL_TO_PASS"),
        PASS_TO_PASS=parse_list("PASS_TO_PASS"),
    )


spec = make_eval_only_spec(record)
client = docker.from_env()
source = (f"xingyaoww/{spec.instance_image_key}").replace("__", "_s_")
try:
    image = client.images.get(source)
except docker.errors.ImageNotFound:
    image = client.images.pull(source)
image.tag(spec.instance_image_key)

# SWE-Gym publishes fully prepared per-instance evaluation images. The fork's
# validator normally derives an instance image from a separate environment
# image, which SWE-Gym does not ship. Keep its official evaluation script and
# reporting path, but use the published immutable evaluation image directly.
def build_direct_eval_container(test_spec, docker_client, run_id, logger, *_args, **_kwargs):
    """Use the official published SWE-Gym eval image without rebuilding it."""
    return docker_client.containers.create(
        image=test_spec.instance_image_key,
        name=test_spec.get_instance_container_name(run_id),
        user="root",
        detach=True,
        command="tail -f /dev/null",
        platform=test_spec.platform,
        mem_limit="16g",
        oom_kill_disable=False,
        network_mode="none",
    )


validation.build_container = build_direct_eval_container

result = {"instance_id": TASK_ID, "repo": record["repo"], "image": source}
for label, patch in (("base", ""), ("gold", record["patch"]), ("reset", "")):
    output = validation.run_instance(
        spec,
        {"instance_id": TASK_ID, "model_name_or_path": f"swegym_{label}", "model_patch": patch},
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=f"swegym_validation_{label}",
        timeout=1200,
    )
    result[label] = output[1][TASK_ID] if output else {"error": "harness_returned_no_report"}

out = ROOT / "results" / f"swegym_validation_{TASK_ID}.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
