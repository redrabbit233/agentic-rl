# SWE-Gym training-source audit

## Decision

The Qwen3 Coding Agent RL training source is `SWE-Gym/SWE-Gym`, not the
SWE-bench train split.  The pinned train split contains 2,438 executable,
repository-level tasks.  `SWE-bench Verified` is excluded from task selection,
training, difficulty selection, checkpoint selection, and internal
evaluation.  It may be used only for a separately run final official
benchmark.

## Pinned sources

- Dataset: `SWE-Gym/SWE-Gym`, `train` split (2,438 records).
- SWE-Gym environment source: `GeneralReasoning/env-swe-gym` at
  `06e10617cbd7b22c2017852ca5bc6dbffe1980b9`.
- Test harness: `SWE-Bench-Fork` at
  `242429c188fcfd06aad13fce9a54d450470bf0ac`.

The source records carry `instance_id`, `repo`, `base_commit`, `version`,
`problem_statement`, `patch`, `test_patch`, `PASS_TO_PASS`, and
`FAIL_TO_PASS`.  Gold patches and test patches are stored only in the ignored
private reference file and are never included in agent prompts or public
manifests.

## Fixed pilot split

`scripts/select_swe_gym_pool.py` uses the deterministic seed
`qwen3-swe-rl-swegym-v1`.  It selects 30 tasks from each of four repos:
`getmoto/moto`, `pydantic/pydantic`, `conan-io/conan`, and `dask/dask`.
The first 25 per repo form the 100-task train pool; the remaining five per
repo form the 20-task internal-dev pool.  The two splits are disjoint.

`data/swe_train_smoke_10.json` is a fixed 10-task verifier gate spanning the
same four repos.  It is not an additional training split.

## Docker verifier gate

Each task starts from the official prebuilt SWE-Gym `sweb.eval` image.  The
official SWE-Bench-Fork evaluation script and report parser are retained.  The
fork's normal image-construction step is skipped because SWE-Gym publishes the
fully prepared per-instance evaluation image rather than the fork's separate
`sweb.env` parent image.

For each task, three independent, no-network containers run:

1. base commit plus hidden `test_patch` must fail;
2. a fresh container plus hidden `test_patch` and the gold patch must pass;
3. another fresh base container plus hidden `test_patch` must fail again.

`network_mode=none` is deliberate: rootless Docker's bridge setup requires an
unavailable namespaced iptables entry point on this host.  It avoids changing
any host firewall rule and prevents test containers from accessing the
network.

Only tasks that complete this `FAIL -> gold PASS -> reset FAIL` gate twice as
needed will be used for Qwen3 base-agent trajectories and the G=4
learnability probe.
