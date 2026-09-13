# Project-local Python environment

Date checked: 2026-09-11

The Coding Agent RL project uses an isolated, user-owned virtual environment:

`/data/chenduo/workspace/agentic_rl/qwen3_swe_rl/.venv`

No system Python files, OS packages, or privileged services were modified.

## Recorded environment

- Python: 3.10.12
- pip: 26.2.1
- site packages: `/data/chenduo/workspace/agentic_rl/qwen3_swe_rl/.venv/lib/python3.10/site-packages`
- Bootstrap: `python3 -m venv --without-pip`, followed by a project-local `get-pip.py` bootstrap because the original venv did not contain pip.
- Packaging tools: pip, setuptools, and wheel are installed inside `.venv` only.
- Agent dependency: `mini-swe-agent[dev]` was installed editable from `third_party/mini-swe-agent` into this venv.

`requirements-lock.txt` must be generated from this venv only once the data-source gate permits the project to proceed.
