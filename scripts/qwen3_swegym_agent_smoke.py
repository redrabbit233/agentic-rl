#!/usr/bin/env python3
"""Hardened SWE-Gym agent runtime; model rollout is intentionally paused."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import docker
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
  lines=p.read_text(errors="replace").splitlines()
  if start>len(lines): raise ValueError("start line is beyond end of file")
  out("\n".join(lines[start-1:end]))
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
 root: Path; model_path: str|None; docker_socket: str|None; max_turns: int; device: str; device_map: str|None; dtype: str; load_in_4bit: bool

@dataclass(frozen=True)
class DecodeConfig:
 do_sample: bool; temperature: float; top_p: float

def config_from_args() -> tuple[Config,tuple]:
 parser=argparse.ArgumentParser(); root=os.environ.get("AGENTIC_RL_ROOT",str(Path(__file__).resolve().parents[1]))
 parser.add_argument("--project-root",default=root); parser.add_argument("--model-path",default=os.environ.get("QWEN_MODEL_PATH")); parser.add_argument("--docker-socket",default=os.environ.get("DOCKER_HOST")); parser.add_argument("--max-turns",type=int,default=int(os.environ.get("AGENT_MAX_TURNS","15"))); parser.add_argument("--device",default=os.environ.get("AGENT_DEVICE","cuda:0")); parser.add_argument("--device-map",default=os.environ.get("AGENT_DEVICE_MAP")); parser.add_argument("--dtype",default=os.environ.get("AGENT_DTYPE","float16"),choices=("float16","bfloat16")); parser.add_argument("--load-in-4bit",action="store_true"); parser.add_argument("--tool-sandbox-smoke",action="store_true"); parser.add_argument("--model-smoke",action="store_true"); parser.add_argument("--run-rollout",action="store_true"); parser.add_argument("--task-ids",default="conan-io__conan-10213"); parser.add_argument("--output",default=None); parser.add_argument("--summary-output",default=None); parser.add_argument("--samples-per-task",type=int,default=1); parser.add_argument("--do-sample",action="store_true"); parser.add_argument("--temperature",type=float,default=0.7); parser.add_argument("--top-p",type=float,default=0.95); parser.add_argument("--seed-base",type=int,default=20260913)
 a=parser.parse_args(); return Config(Path(a.project_root).resolve(),a.model_path,a.docker_socket,a.max_turns,a.device,a.device_map,a.dtype,a.load_in_4bit),(a.tool_sandbox_smoke,a.model_smoke,a.run_rollout,[x for x in a.task_ids.split(",") if x],a.output,a.summary_output,a.samples_per_task,DecodeConfig(a.do_sample,a.temperature,a.top_p),a.seed_base)

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
 def final_patch(self)->str:
  return self.execv(["git","diff"],30)[1]
 def close(self):
  try:self.container.remove(force=True)
  except Exception:pass

def sandbox_smoke(cfg:Config)->None:
 private={x["instance_id"]:x for x in json.loads((cfg.root/".private"/"swe_gym_reference_records.json").read_text())}; validated={x["instance_id"] for x in json.loads((cfg.root/"data"/"swegym_validated_tasks.json").read_text())["tasks"]}; task=next(iter(validated)); s=TaskContainer(private[task],cfg,validated)
 try:
  listed,list_output,_=s.call("list_files",{"path":"."}); first_file=next((x for x in list_output.splitlines() if x),None)
  checks={"list_root_rejected":not s.call("list_files",{"path":"/"})[0],"search_parent_rejected":not s.call("search_text",{"path":"../","query":"x"})[0],"tmp_read_rejected":not s.call("read_file",{"path":"/tmp/.hidden_verifier.sh"})[0],"command_substitution_literal":s.call("search_text",{"path":".","query":"$(touch /tmp/pwned)"})[0],"normal_list":listed,"normal_search":s.call("search_text",{"path":".","query":"def"})[0],"normal_read":bool(first_file) and s.call("read_file",{"path":first_file,"start":1,"end":5})[0]}
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

SYSTEM="""You are a coding agent. Use only one JSON object per turn: {\"tool\":name,\"arguments\":object}.
All paths are repository-relative. Tools: list_files({\"path\":\".\"}), search_text({\"query\":\"...\",\"path\":\".\"}), read_file({\"path\":\"...\",\"start\":1,\"end\":160}), apply_patch({\"path\":\"...\",\"find\":\"unique old text\",\"replace\":\"new text\"}), run_tests({}), final({\"summary\":\"...\"}). Start with search/read, edit source (never tests), then run_tests. If tests fail, use feedback before retrying."""

def parse_tool(raw:str)->dict|None:
 decoder=json.JSONDecoder(); candidates=[]; cleaned=raw.replace("<tool_call>","").replace("</tool_call>","")
 for index,char in enumerate(cleaned):
  if char!="{":continue
  try:
   value,_=decoder.raw_decode(cleaned[index:])
   if isinstance(value,dict) and isinstance(value.get("tool"),str) and isinstance(value.get("arguments",{}),dict):candidates.append(value)
  except json.JSONDecodeError:pass
 return candidates[-1] if candidates else None

def model_input_device(model):
 mapping=getattr(model,"hf_device_map",{})
 for location in mapping.values():
  if isinstance(location,int): return torch.device(f"cuda:{location}")
  if isinstance(location,str) and location.startswith("cuda:"): return torch.device(location)
 return model.device

def generate(model,tokenizer,messages:list[dict],decode:DecodeConfig)->tuple[str,int]:
 prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 encoded=tokenizer(prompt,return_tensors="pt").to(model_input_device(model))
 kwargs={"max_new_tokens":384,"do_sample":decode.do_sample,"pad_token_id":tokenizer.eos_token_id}
 if decode.do_sample: kwargs.update({"temperature":decode.temperature,"top_p":decode.top_p})
 with torch.inference_mode(): output=model.generate(**encoded,**kwargs)
 tokens=output[0][encoded.input_ids.shape[1]:]
 return tokenizer.decode(tokens,skip_special_tokens=True),int(tokens.shape[0])

def run_task(model,tokenizer,public:dict,private:dict,cfg:Config,validated:set[str],decode:DecodeConfig,sample_index:int,seed:int)->dict:
 torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 started=time.monotonic(); session=TaskContainer(private,cfg,validated); messages=[{"role":"system","content":SYSTEM},{"role":"user","content":"Issue:\n"+public["problem_statement"]}]
 calls=[]; outputs=[]; patches=[]; tests=[]; total_tokens=0; invalid=0; final_called=False; termination="MAX_TURNS"; reward=None; hidden=None
 try:
  for turn in range(1,cfg.max_turns+1):
   raw,tokens=generate(model,tokenizer,messages,decode); total_tokens+=tokens; call=parse_tool(raw)
   if not call:
    invalid+=1; calls.append({"turn":turn,"parse_success":False,"raw":raw}); messages.extend([{"role":"assistant","content":raw},{"role":"user","content":"Invalid tool JSON. Output exactly one JSON object."}])
    if invalid>=3:termination="INVALID_TOOL_CALL";break
    continue
   name,args=call["tool"],call["arguments"]; calls.append({"turn":turn,"parse_success":True,"tool":name,"arguments":args}); messages.append({"role":"assistant","content":json.dumps(call)})
   if name=="final": final_called=True; break
   ok,observation,meta=session.call(name,args); outputs.append({"turn":turn,"tool":name,"success":ok,"output":observation,**meta})
   if name=="apply_patch":patches.append({"turn":turn,"success":ok,"edit":args})
   if name=="run_tests":tests.append({"turn":turn,"success":ok,"output":observation,**meta})
   messages.append({"role":"user","content":f"Tool {name} result success={ok}:\n{observation}"})
  reward,hidden=session.final_hidden_verifier(); termination=classify_termination(final_called,reward,termination)
  return {"instance_id":public["instance_id"],"sample_index":sample_index,"seed":seed,"messages":messages,"tool_calls":calls,"tool_outputs":outputs,"patch_history":patches,"test_history":tests,"turns":len(calls),"generated_tokens":total_tokens,"wall_time":round(time.monotonic()-started,2),"final_patch":session.final_patch(),"hidden_verifier_result":hidden,"reward":int(reward),"termination_reason":termination}
 except Exception as exc:
  return {"instance_id":public["instance_id"],"sample_index":sample_index,"seed":seed,"messages":messages,"tool_calls":calls,"tool_outputs":outputs,"patch_history":patches,"test_history":tests,"turns":len(calls),"generated_tokens":total_tokens,"wall_time":round(time.monotonic()-started,2),"final_patch":"","hidden_verifier_result":None,"reward":None,"termination_reason":"ENV_ERROR","error":repr(exc)}
 finally:session.close()

def load_model(cfg:Config):
 dtype={"float16":torch.float16,"bfloat16":torch.bfloat16}[cfg.dtype]
 kwargs={"trust_remote_code":True,"device_map":cfg.device_map or cfg.device}
 if cfg.load_in_4bit:
  kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=dtype)
 else: kwargs["torch_dtype"]=dtype
 tokenizer=AutoTokenizer.from_pretrained(cfg.model_path,trust_remote_code=True)
 return AutoModelForCausalLM.from_pretrained(cfg.model_path,**kwargs).eval(),tokenizer

def main():
 cfg,options=config_from_args(); smoke,model_smoke,run_rollout,task_ids,output_path,summary_path,samples,decode,seed_base=options
 if smoke:sandbox_smoke(cfg);return
 if not cfg.model_path:raise SystemExit("--model-path or QWEN_MODEL_PATH required")
 if not run_rollout and not model_smoke:raise SystemExit("pass --model-smoke or --run-rollout after sandbox approval")
 model,tokenizer=load_model(cfg)
 if model_smoke:
  text,tokens=generate(model,tokenizer,[{"role":"user","content":"Return exactly: model-ready"}],DecodeConfig(False,0.7,0.95))
  print(json.dumps({"model_smoke_output":text,"generated_tokens":tokens,"device_map":getattr(model,"hf_device_map",{}),"dtype":cfg.dtype,"load_in_4bit":cfg.load_in_4bit}))
  if not run_rollout:return
 validated={x["instance_id"] for x in json.loads((cfg.root/"data"/"swegym_validated_tasks.json").read_text())["tasks"]}
 if any(x not in validated for x in task_ids):raise SystemExit("all task ids must be in swegym_validated_tasks.json")
 public={x["instance_id"]:x for x in json.loads((cfg.root/"data"/"swe_gym_train_100.json").read_text())}
 private={x["instance_id"]:x for x in json.loads((cfg.root/".private"/"swe_gym_reference_records.json").read_text())}
 missing=[x for x in task_ids if x not in public or x not in private]
 if missing:raise SystemExit(f"task records unavailable: {missing}")
 out=Path(output_path) if output_path else cfg.root/"trajectories"/"qwen3_base_smoke.jsonl"; out.parent.mkdir(parents=True,exist_ok=True)
 results=[]
 with out.open("w") as handle:
  for task_id in task_ids:
   for sample_index in range(samples):
    trajectory_index=len(results); seed=seed_base+trajectory_index
    result=run_task(model,tokenizer,public[task_id],private[task_id],cfg,validated,decode,sample_index,seed); results.append(result); handle.write(json.dumps(result)+"\n"); handle.flush(); print(json.dumps({k:result.get(k) for k in ("instance_id","sample_index","seed","reward","turns","generated_tokens","wall_time","termination_reason")}))
 if summary_path:
  vectors={task_id:[r["reward"] for r in results if r["instance_id"]==task_id] for task_id in task_ids}
  groups=list(vectors.values()); summary={"model_path":cfg.model_path,"task_ids":task_ids,"samples_per_task":samples,"seed_base":seed_base,"decode":{"do_sample":decode.do_sample,"temperature":decode.temperature,"top_p":decode.top_p},"total_trajectories":len(results),"positive_trajectories":sum(r["reward"]==1 for r in results),"reward_vectors":vectors,"mixed_groups":sum(0 in x and 1 in x for x in groups),"all_fail_groups":sum(all(v==0 for v in x) for x in groups),"all_pass_groups":sum(all(v==1 for v in x) for x in groups),"avg_turns":sum(r["turns"] for r in results)/len(results),"avg_generated_tokens":sum(r["generated_tokens"] for r in results)/len(results),"avg_rollout_seconds":sum(r["wall_time"] for r in results)/len(results),"valid_patch_tasks":sum(bool(r["patch_history"]) for r in results),"tool_parse_success":sum(c["parse_success"] for r in results for c in r["tool_calls"]),"tool_calls":sum(len(r["tool_calls"]) for r in results),"tool_execution_success":sum(o["success"] for r in results for o in r["tool_outputs"]),"tool_executions":sum(len(r["tool_outputs"]) for r in results),"pytest_calls":sum(len(r["test_history"]) for r in results)}; Path(summary_path).write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__":main()
