# Qwen3 SWE RL experiment environment

Captured: 2026-09-13

## Rootless Docker

- Docker client/server: 29.8.0.
- Rootless daemon owner: `chenduo`.
- API socket: `/run/user/1005/docker.sock`.
- Data root: `/data/chenduo/docker-rootless`.
- Storage driver: `overlayfs`.
- No root `dockerd`, `/var/run/docker.sock`, or Docker TCP listener was used.
- Official SWE-bench harness image smoke: `pallets__flask-4045` built successfully.

## Harness and model runtime

- SWE-bench harness checkout: tag `v2.1.7`, commit `5f5a7df799663adba4b191eca3d675faf3621fe2`.
- Model: `Qwen3-4B-Instruct-2507` at
  `/data/chenduo/workspace/agentic_rl/qwen3_agentic_rl/models/Qwen3-4B-Instruct-2507`.
- Model config SHA-256: `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`.
- Python: 3.10.12; PyTorch: 2.7.0+cu126; Transformers: 4.53.3;
  PEFT: 0.16.0; Accelerate: 1.8.1; vLLM: not installed.

## Dataset provenance gate

The legacy official `princeton-nlp/SWE-bench` revision
`e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6` exposes a 19,008-record train
split, but all inspected train records have empty `test_patch`, `version`, and
`FAIL_TO_PASS`. The official `princeton-nlp/SWE-bench_oracle` train split
(18,817 records) has the same limitation. Conversely, the current official
`SWE-bench/SWE-bench` package exposes only `dev` and `test` splits.

Therefore a strict train-only verifier requiring the dataset-provided
`test_patch` cannot yet be constructed from these official sources. Test,
Verified, and dev instances remain excluded from training, prompt tuning,
difficulty selection, and checkpoint selection.

No model weights, Docker images, caches, private reference records, or secrets
belong in version control.
