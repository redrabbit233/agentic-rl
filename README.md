# Qwen3 SWE-Gym Coding Agent RL

This repository contains the reproducible public infrastructure for a
Qwen3-4B coding-agent reinforcement-learning pilot using `SWE-Gym/SWE-Gym`.

The current sequence is deliberately gated:

1. select deterministic, disjoint 100-task train and 20-task internal-dev pools;
2. validate executable Docker tasks with `FAIL -> gold PASS -> reset FAIL`;
3. run base-agent tool-use smoke trajectories;
4. measure an internal-dev base baseline and a G=4 learnability probe;
5. decide whether GRPO is justified.

SWE-bench Verified is not included in training, task selection, model
selection, or internal evaluation. It is reserved for a later official final
benchmark.

## Privacy and data boundaries

Public manifests include task IDs, provenance, commits, issue statements, and
counts only. Reference patches and test patches are deliberately excluded from
Git, prompts, trajectories, and training data. Prebuilt upstream repository
snapshots, Docker storage, model weights, and private evaluator records are
also excluded.

## Layout

- `scripts/select_swe_gym_pool.py` — deterministic pilot-pool selection.
- `scripts/validate_one_swe_gym_task.py` — isolated three-stage verifier gate.
- `scripts/qwen3_swegym_agent_smoke.py` — Qwen3 tool-use smoke controller.
- `data/swegym_validated_tasks.json` — public verifier evidence without patch
  contents.
- `docs/SWE_GYM_DATA_AUDIT.md` — source, split, and verifier policy.

See the documentation for environment requirements and exact safety
constraints before executing the scripts.
