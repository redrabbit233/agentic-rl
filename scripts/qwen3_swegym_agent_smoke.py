#!/usr/bin/env python3
"""Run Qwen3 tool-use smoke trajectories in isolated SWE-Gym containers.

The controller intentionally never places reference patches or test-patch text
in model messages.  Hidden verifier scripts are copied into a container only
when `run_tests` is invoked and their source is never returned to the model.
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
import time
import uuid
from pathlib import Path

import docker
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.run_validation import get_validation_report
from swebench.harness.test_spec import TestSpec, make_eval_script_list

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = "/data/chenduo/workspace/agentic_rl/qwen3_agentic_rl/models/Qwen3-4B-Instruct-2507"
MAX_TURNS = 15
SMOKE_IDS = [
    "getmoto__moto-6785",
    "getmoto__moto-6953",
    "pydantic__pydantic-5529",
    "conan-io__conan-10213",
    "dask__dask-7418",
]
SYSTEM = """You are a coding agent repairing one repository issue. Work only through tools.
Never claim a test passed unless run_tests reports it. Do not edit tests, do not access the
network, and do not use shell commands outside the supplied tools. Respond with exactly one
JSON object and nothing else. Schema: {\"tool\": TOOL_NAME, \"arguments\": OBJECT}.
Tools: list_files({\"path\":\".\"}), search_text({\"query\":\"...\",\"path\":\".\"}),
read_file({\"path\":\"...\",\"start\":1,\"end\":160}), apply_patch({\"patch\":\"unified diff\"}),
run_tests({}), final({\"summary\":\"...\"}). Start by searching and reading relevant code,
then edit and test. If tests fail, inspect feedback and make a revised patch."""


def eval_only_spec(instance: dict) -> TestSpec:
    repo = instance["repo"].lower()
    specs = MAP_REPO_VERSION_TO_SPECS[repo][instance["version"]]
    to_list = lambda k: json.loads(instance[k]) if isinstance(instance[k], str) else instance[k]
    return TestSpec(
        instance_id=instance["instance_id"].lower(), repo=repo, version=instance["version"],
        repo_script_list=[], env_script_list=[], arch="x86_64",
        eval_script_list=make_eval_script_list(
            instance, specs, "testbed", "/testbed", instance["base_commit"], instance["test_patch"]
        ), FAIL_TO_PASS=to_list("FAIL_TO_PASS"), PASS_TO_PASS=to_list("PASS_TO_PASS"),
    )


def put_text(container, text: str, target: str) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        raw = text.encode()
        info = tarfile.TarInfo(Path(target).name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    data.seek(0)
    container.put_archive(str(Path(target).parent), data.getvalue())


def clipped(value: bytes | str, limit: int = 6000) -> str:
    value = value.decode(errors="replace") if isinstance(value, bytes) else value
    return value[-limit:] if len(value) > limit else value


class TaskContainer:
    def __init__(self, task: dict):
        self.task = task
        self.spec = eval_only_spec(task)
        self.client = docker.from_env()
        self.container = self.client.containers.create(
            image=f"sweb.eval.x86_64.{self.spec.instance_id}:latest",
            name=f"qwen3-smoke-{self.spec.instance_id}-{uuid.uuid4().hex[:8]}",
            user="root", detach=True, command="tail -f /dev/null", platform=self.spec.platform,
            mem_limit="16g", oom_kill_disable=False, network_mode="none",
        )
        self.container.start()

    def exec(self, command: str, timeout: int = 180) -> tuple[int, str]:
        result = self.container.exec_run(f"timeout {timeout}s /bin/bash -lc {json.dumps(command)}", workdir="/testbed", user="root")
        return result.exit_code, clipped(result.output)

    def run_hidden_verifier(self) -> tuple[bool, str, dict]:
        """Run hidden tests and grade exactly as the official harness does."""
        put_text(self.container, self.spec.eval_script, "/tmp/.hidden_verifier.sh")
        exit_code, output = self.exec("/bin/bash /tmp/.hidden_verifier.sh", timeout=300)
        log_dir = ROOT / "logs" / "agent_hidden_verifier" / self.spec.instance_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{uuid.uuid4().hex}.txt"
        log_path.write_text(output)
        report = get_validation_report(
            test_spec=self.spec,
            prediction={"instance_id": self.spec.instance_id, "model_patch": ""},
            log_path=log_path,
            include_tests_status=False,
        )[self.spec.instance_id]
        resolved = bool(report["resolved"])
        # Do not expose generated verifier scripts or patch text to the model.
        return resolved, f"Verifier resolved={resolved}; process_exit={exit_code}.\n{output}", {
            "process_exit_code": exit_code, "resolved": resolved, "verifier_log": str(log_path.relative_to(ROOT)),
        }

    def call(self, name: str, args: dict) -> tuple[bool, str, dict]:
        if name == "list_files":
            path = args.get("path", ".")
            code, output = self.exec(f"find {json.dumps(path)} -maxdepth 3 -type f | head -200")
        elif name == "search_text":
            query, path = args.get("query"), args.get("path", ".")
            if not isinstance(query, str) or not query:
                return False, "query must be a nonempty string", {}
            code, output = self.exec(f"grep -RIn --exclude-dir=.git --exclude='*.pyc' -- {json.dumps(query)} {json.dumps(path)} | head -120")
        elif name == "read_file":
            path = args.get("path")
            if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
                return False, "path must be a relative repository file", {}
            start, end = int(args.get("start", 1)), int(args.get("end", 160))
            code, output = self.exec(f"sed -n {json.dumps(str(max(1,start)) + ',' + str(min(start+400,end)) + 'p')} -- {json.dumps(path)}")
        elif name == "apply_patch":
            patch = args.get("patch")
            if not isinstance(patch, str) or not patch.strip():
                return False, "patch must be a nonempty unified diff", {}
            put_text(self.container, patch, "/tmp/agent.patch")
            code, output = self.exec("git apply --whitespace=nowarn /tmp/agent.patch; git diff --stat")
        elif name == "run_tests":
            return self.run_hidden_verifier()
        else:
            return False, f"unknown tool: {name}", {}
        return code == 0, output, {"exit_code": code}

    def final_patch(self) -> str:
        _code, output = self.exec("git diff", timeout=30)
        return output

    def close(self) -> None:
        try:
            self.container.remove(force=True)
        except Exception:
            pass


def parse_tool(raw: str) -> dict | None:
    raw = raw.replace("<tool_call>", "").replace("</tool_call>", "")
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
            if isinstance(value, dict) and isinstance(value.get("tool"), str):
                candidates.append(value)
        except json.JSONDecodeError:
            continue
    return candidates[-1] if candidates else None


def generate(model, tokenizer, messages: list[dict]) -> tuple[str, int]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = output[0][encoded.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.shape[0])


def run_task(model, tokenizer, public: dict, private: dict) -> dict:
    start = time.monotonic()
    session = TaskContainer(private)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Issue:\n" + public["problem_statement"]}]
    tool_calls, patch_history, test_history = [], [], []
    token_count = 0
    invalid = 0
    termination = "MAX_TURNS"
    try:
        for turn in range(1, MAX_TURNS + 1):
            raw, tokens = generate(model, tokenizer, messages)
            token_count += tokens
            call = parse_tool(raw)
            if not call or not isinstance(call.get("arguments", {}), dict):
                invalid += 1
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Invalid tool JSON. Return exactly one valid tool JSON object."})
                tool_calls.append({"turn": turn, "raw": raw, "parse_success": False})
                if invalid >= 3:
                    termination = "INVALID_TOOL_CALL"
                    break
                continue
            name, args = call["tool"], call["arguments"]
            tool_calls.append({"turn": turn, "tool": name, "arguments": args, "parse_success": True})
            messages.append({"role": "assistant", "content": json.dumps(call)})
            if name == "final":
                termination = "AGENT_ABORT"
                break
            ok, observation, meta = session.call(name, args)
            if name == "apply_patch":
                patch_history.append({"turn": turn, "ok": ok, "patch": args.get("patch", ""), "output": observation})
            if name == "run_tests":
                test_history.append({"turn": turn, "ok": ok, "output": observation, **meta})
            messages.append({"role": "user", "content": f"Tool {name} result (success={ok}):\n{observation}"})
        # A hidden verifier run is always performed, even when the agent quits early.
        verified, final_output, meta = session.call("run_tests", {})
        test_history.append({"turn": "final", "ok": verified, "output": final_output, **meta})
        if verified:
            termination = "SOLVED"
        elif termination == "MAX_TURNS" and any(x["turn"] != "final" for x in test_history):
            termination = "MAX_TURNS"
        elif termination == "MAX_TURNS":
            termination = "AGENT_ABORT"
        return {
            "instance_id": public["instance_id"], "messages": messages, "tool_calls": tool_calls,
            "tool_outputs": [{"tool": "run_tests_final", "success": verified, "output": final_output}],
            "patch_history": patch_history, "test_history": test_history, "turns": len(tool_calls),
            "generated_tokens": token_count, "wall_time": round(time.monotonic() - start, 2),
            "final_patch": session.final_patch(), "verifier_result": verified, "success": verified,
            "termination_reason": termination,
        }
    except Exception as exc:
        return {"instance_id": public["instance_id"], "messages": messages, "tool_calls": tool_calls,
                "patch_history": patch_history, "test_history": test_history, "turns": len(tool_calls),
                "generated_tokens": token_count, "wall_time": round(time.monotonic() - start, 2),
                "final_patch": "", "verifier_result": None, "success": None,
                "termination_reason": "ENV_ERROR", "error": repr(exc)}
    finally:
        session.close()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    public_records = {x["instance_id"]: x for x in json.loads((ROOT / "data" / "swe_gym_train_100.json").read_text())}
    private_records = {x["instance_id"]: x for x in json.loads((ROOT / ".private" / "swe_gym_reference_records.json").read_text())}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True).eval()
    out = ROOT / "trajectories" / "qwen3_base_smoke.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as handle:
        for task_id in SMOKE_IDS:
            result = run_task(model, tokenizer, public_records[task_id], private_records[task_id])
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(json.dumps({k: result.get(k) for k in ("instance_id", "success", "turns", "generated_tokens", "wall_time", "termination_reason")}))


if __name__ == "__main__":
    main()
