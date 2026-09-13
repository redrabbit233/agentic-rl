#!/usr/bin/env bash
# Run the nine remaining fixed SWE-Gym Docker validation tasks sequentially.
# The first task is validated separately before this batch starts.
set -uo pipefail

ROOT="/data/chenduo/workspace/agentic_rl/qwen3_swe_rl"
export DOCKER_HOST="unix:///run/user/1005/docker.sock"
cd "$ROOT"

tasks=(
  getmoto__moto-4799
  getmoto__moto-6953
  pydantic__pydantic-5744
  pydantic__pydantic-5529
  pydantic__pydantic-5591
  conan-io__conan-11799
  conan-io__conan-10213
  dask__dask-7418
  dask__dask-6749
)

for task in "${tasks[@]}"; do
  echo "[$(date -Is)] START $task"
  if .venv/bin/python scripts/validate_one_swe_gym_task.py "$task"; then
    echo "[$(date -Is)] DONE $task"
  else
    echo "[$(date -Is)] SCRIPT_ERROR $task"
  fi
done
