#!/usr/bin/env python3
"""Hardened SWE-Gym agent runtime; model rollout is intentionally paused."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import docker
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.run_validation import get_validation_report
from swebench.harness.test_spec import TestSpec, make_eval_script_list

TOOL_SERVER = r'''import json,sys
from pathlib import Path,PurePosixPath
ROOT=Path("/testbed").resolve()
def safe(raw):
 if not isinstance(raw,str) or not raw or PurePosixPath(raw).is_absolute() or ".." in PurePosixPath(raw).parts: raise ValueError("path must be a non-absolute repository-relative path without '..'")
 p=(ROOT/raw).resolve()
 if p!=ROOT and ROOT not in p.parents: raise ValueError("resolved path escapes repository root")
 return p
def out(x): print(x[:12000])
action,arg=sys.argv[1],json.loads(sys.argv[2])
try:
 if action=="list_files":
  p=safe(arg.get("path",".")); files=[str(x.relative_to(ROOT)) for x in p.rglob("*") if x.is_file() and ".git" not in x.parts]; out("\n".join(files[:200]))
 elif action=="search_text":
  p,q=safe(arg.get("path",".")),arg.get("query")
  if not isinstance(q,str) or not q: raise ValueError("query must be a nonempty string")
  found=[]
  for x in p.rglob("*") if p.is_dir() else [p]:
   if not x.is_file() or ".git" in x.parts: continue
   try:
    for n,line in enumerate(x.read_text(errors="replace").splitlines(),1):
     if q in line: found.append(f"{x.relative_to(ROOT)}:{n}:{line}")
     if len(found)>=120: break
   except (OSError,UnicodeError): pass
   if len(found)>=120: break
  out("\n".join(found))
 elif action=="read_file":
  p=safe(arg.get("path")); start=max(1,int(arg.get("start",1))); end=min(start+400,max(start,int(arg.get("end",160))))
  if not p.is_file(): raise ValueError("path is not a regular file")
  out("\n".join(p.read_text(errors="replace").splitlines()[start-1:end]))
 elif action=="apply_patch":
  p,old,new=safe(arg.get("path")),arg.get("find"),arg.get("replace")
  if not p.is_file() or not isinstance(old,str) or not old or not isinstance(new,str): raise ValueError("need file path plus nonempty find and replace strings")
  text=p.read_text(); count=text.count(old)
  if count!=1: raise ValueError(f"find text occurs {count} times; choose a unique string")
  p.write_text(text.replace(old,new,1)); out("edit applied")
 else: raise ValueError("unknown tool action")
except Exception as e: print("TOOL_ERROR: "+str(e)); sys.exit(2)
'''

@dataclass(frozen=True)
class Config:
 root: Path; model_path: str|None; docker_socket: str|None; max_turns: int; device: str

def config_from_args() -> tuple[Config,bool]:
 parser=argparse.ArgumentParser(); root=os.environ.get("AGENTIC_RL_ROOT",str(Path(__file__).resolve().parents[1]))
 parser.add_argument("--project-root",default=root); parser.add_argument("--model-path",default=os.environ.get("QWEN_MODEL_PATH")); parser.add_argument("--docker-socket",default=os.environ.get("DOCKER_HOST")); parser.add_argument("--max-turns",type=int,default=int(os.environ.get("AGENT_MAX_TURNS","15"))); parser.add_argument("--device",default=os.environ.get("AGENT_DEVICE","cuda:0")); parser.add_argument("--tool-sandbox-smoke",action="store_true")
 a=parser.parse_args(); return Config(Path(a.project_root).resolve(),a.model_path,a.docker_socket,a.max_turns,a.device),a.tool_sandbox_smoke

def spec_for(instance:dict)->TestSpec:
 repo=instance["repo"].lower(); specs=MAP_REPO_VERSION_TO_SPECS[repo][instance["version"]]
 as_list=lambda k: json.loads(instance[k]) if isinstance(instance[k],str) else instance[k]
 return TestSpec(instance_id=instance["instance_id"].lower(),repo=repo,version=instance["version"],repo_script_list=[],env_script_list=[],arch="x86_64",eval_script_list=make_eval_script_list(instance,specs,"testbed","/testbed",instance["base_commit"],instance["test_patch"]),FAIL_TO_PASS=as_list("FAIL_TO_PASS"),PASS_TO_PASS=as_list("PASS_TO_PASS"))

def put_text(container,text:str,target:str)->None:
 data=io.BytesIO()
 with tarfile.open(fileobj=data,mode="w") as tar:
  raw=text.encode(); info=tarfile.TarInfo(Path(target).name); info.size=len(raw); tar.addfile(info,io.BytesIO(raw))
 container.put_archive(str(Path(target).parent),data.getvalue())

class TaskContainer:
 def __init__(self,task:dict,cfg:Config,validated:set[str]):
  if task["instance_id"] not in validated: raise ValueError("instance_id is not validated")
  self.cfg,self.spec=cfg,spec_for(task); self.client=docker.DockerClient(base_url=cfg.docker_socket) if cfg.docker_socket else docker.from_env()
  self.container=self.client.containers.create(image=f"sweb.eval.x86_64.{self.spec.instance_id}:latest",name=f"qwen3-agent-{uuid.uuid4().hex[:10]}",user="root",detach=True,command=["tail","-f","/dev/null"],platform=self.spec.platform,mem_limit="16g",oom_kill_disable=False,network_mode="none")
  self.container.start(); self.tool_path=f"/tmp/.agent-tool-{uuid.uuid4().hex}.py"; put_text(self.container,TOOL_SERVER,self.tool_path)
 def execv(self,args:list[str],timeout:int=180)->tuple[int,str]:
  r=self.container.exec_run(["timeout",str(timeout),*args],workdir="/testbed",user="root"); text=r.output.decode(errors="replace"); return r.exit_code,text[-6000:]
 def call(self,name:str,args:dict)->tuple[bool,str,dict]:
  if name in {"list_files","search_text","read_file","apply_patch"}: code,out=self.execv(["python",self.tool_path,name,json.dumps(args)])
  elif name=="run_tests": code,out=self.execv(["python","-m","pytest","-q"],180)
  else: return False,"unknown tool",{}
  return code==0,out,{"exit_code":code}
 def final_hidden_verifier(self)->tuple[bool,dict]:
  path=f"/tmp/.hidden-verifier-{uuid.uuid4().hex}.sh"
  try:
   put_text(self.container,self.spec.eval_script,path); code,out=self.execv(["/bin/bash",path],300)
   logs=self.cfg.root/"logs"/"agent_hidden_verifier"/self.spec.instance_id; logs.mkdir(parents=True,exist_ok=True); log=logs/f"{uuid.uuid4().hex}.txt"; log.write_text(out)
   report=get_validation_report(test_spec=self.spec,prediction={"instance_id":self.spec.instance_id,"model_patch":""},log_path=log,include_tests_status=False)[self.spec.instance_id]
   return bool(report["resolved"]),{"process_exit_code":code,"resolved":bool(report["resolved"]),"verifier_log":str(log.relative_to(self.cfg.root))}
  finally: self.execv(["rm","-f",path],20)
 def close(self):
  try:self.container.remove(force=True)
  except Exception:pass

def sandbox_smoke(cfg:Config)->None:
 private={x["instance_id"]:x for x in json.loads((cfg.root/".private"/"swe_gym_reference_records.json").read_text())}; validated={x["instance_id"] for x in json.loads((cfg.root/"data"/"swegym_validated_tasks.json").read_text())["tasks"]}; task=next(iter(validated)); s=TaskContainer(private[task],cfg,validated)
 try:
  checks={"list_root_rejected":not s.call("list_files",{"path":"/"})[0],"search_parent_rejected":not s.call("search_text",{"path":"../","query":"x"})[0],"tmp_read_rejected":not s.call("read_file",{"path":"/tmp/.hidden_verifier.sh"})[0],"command_substitution_literal":s.call("search_text",{"path":".","query":"$(touch /tmp/pwned)"})[0],"normal_list":s.call("list_files",{"path":"."})[0],"normal_search":s.call("search_text",{"path":".","query":"def"})[0],"normal_read":s.call("read_file",{"path":"README.md","start":1,"end":5})[0]}
  checks["substitution_not_executed"]=s.execv(["test","!","-e","/tmp/pwned"],20)[0]==0; print(json.dumps(checks,indent=2)); assert all(checks.values()),checks
  s.final_hidden_verifier()
  checks["hidden_verifier_removed"]=s.execv(["python","-c","from pathlib import Path; import sys; sys.exit(any(Path('/tmp').glob('.hidden-verifier-*.sh')) )"],20)[0]==0
  print(json.dumps(checks,indent=2)); assert all(checks.values()),checks
 finally:s.close()

def classify_termination(final_called:bool, verifier_pass:bool, prior:str="MAX_TURNS")->str:
 """Use this when rollout is re-enabled after the hardened smoke gate."""
 if verifier_pass:return "SOLVED"
 if final_called:return "FINAL_UNSOLVED"
 return prior

def main():
 cfg,smoke=config_from_args()
 if smoke:sandbox_smoke(cfg);return
 if not cfg.model_path:raise SystemExit("--model-path or QWEN_MODEL_PATH required")
 raise SystemExit("Model rollout paused pending hardened smoke approval")
if __name__=="__main__":main()
