# SWE-Gym OpenHands baseline pool v2 audit

## Scope

This audit replaces the initial OpenHands baseline sanity pool. It is not an
unbiased SWE-Gym benchmark sample and must not be reported as a SWE-Gym score.
It is a small engineering gate for checking whether the OpenHands-7B policy can
produce valid repository-level trajectories before any G=4 or RL experiment.

Pinned input:

- dataset: `SWE-Gym/SWE-Gym`, split `train`
- local dataset fingerprint: `3c18e7f6e3af6f68`
- candidate source: `data/swegym_validated_tasks.json`
- required verifier state: base FAIL, gold PASS, reset FAIL

Gold and test-patch metadata were inspected only to detect corrupt/misaligned
records and to keep this engineering pilot small. They are never included in
the agent prompt or trajectory. This selection is therefore deliberately
biased and is unsuitable for benchmark reporting.

## Findings

### Excluded: `dask__dask-7418`

The record is internally inconsistent. Its problem statement requests a change
to HDFS CI trigger behavior, while its gold and verifier artifacts target Dask
bag text-reading behavior. An agent following the supplied problem statement
cannot be fairly graded by that verifier. The task is excluded from all
baseline, sampling, and RL gates unless the upstream record is corrected.

### Excluded from v2: Conan tasks

- `conan-io__conan-10213` already received G=1 and G=8 evaluation and produced
  no positive trajectory. Reusing it would not test task diversity.
- `conan-io__conan-11799` is another CMake/Android NDK build-system task and is
  outside the requested lightweight non-build-system sanity set.

### Deferred or reserve

- `pydantic__pydantic-5744` is semantically aligned but has a comparatively
  large multi-file change and broad verifier surface; defer it until the basic
  agent loop is reliable.
- `getmoto__moto-6953` is semantically aligned and remains a reserve task, but
  its change spans more source files and verifier targets than the selected
  pilot tasks.

## Selected v2 pool

| instance | rationale | verifier gate |
| --- | --- | --- |
| `pydantic__pydantic-5529` | Small, semantically aligned validator/decorator change and narrow verifier surface | FAIL → gold PASS → reset FAIL |
| `pydantic__pydantic-5591` | Semantically aligned, single-source-file enum/Literal change with one fail-to-pass target | FAIL → gold PASS → reset FAIL |
| `getmoto__moto-6785` | Semantically aligned authentication failure with a modest two-file source change and one fail-to-pass target | FAIL → gold PASS → reset FAIL |

The public, patch-free manifest is
`data/swegym_openhands_baseline_pool_v2.json`. Detailed patch/test metadata and
hashes are stored only under `.private/` and are ignored by Git.

All three selected tasks are members of the fixed 100-task training split.
They have zero overlap with the fixed 20-task internal-dev split and zero
overlap with the instance IDs in the published OpenHands-7B SFT trajectory
audit. They are therefore suitable for a training-side policy-support sanity,
but not for held-out accuracy reporting.

## Mandatory launch guards

The pinned OpenHands fork does not honor `--eval-ids` for dataset selection. It
reads `evaluation/swe_bench/config.toml` instead. Every future launch must:

1. write exactly one selected ID from the v2 manifest to that selector;
2. verify the log line `Starting evaluation for instance ...` matches it;
3. verify `SWE_INSTANCE_ID`, workspace repo/version, and base commit;
4. treat any mismatch as `ENV_ERROR`, never reward 0;
5. restore the selector after the run, including on interruption.

No G=4 or GRPO run is allowed until at least one selected v2 task completes a
clean G=1 trajectory and hidden grading without inference or environment error.
