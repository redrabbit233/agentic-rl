#!/usr/bin/env python3
"""Minimal loopback-only OpenAI chat endpoint for a local causal LM.

This is the fallback when vLLM cannot start on the host runtime.  It is
deliberately single-request and intended for reproducible agent rollouts, not
general multi-tenant serving.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path: str, tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True
    ).eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--served-model-name", default="openhands7b")
    args = parser.parse_args()
    model, tokenizer = load(args.model_path, args.tokenizer_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def respond(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                self.respond(200, {"object": "list", "data": [{"id": args.served_model_name, "object": "model"}]})
            else:
                self.respond(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": {"message": "not found"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = body["messages"]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                do_sample = bool(body.get("temperature", 0) and body.get("temperature", 0) > 0)
                kwargs = {"max_new_tokens": min(int(body.get("max_tokens", 1024)), 2048), "do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
                if do_sample:
                    kwargs.update({"temperature": float(body["temperature"]), "top_p": float(body.get("top_p", 1.0))})
                with torch.inference_mode():
                    output = model.generate(**encoded, **kwargs)
                tokens = output[0][encoded.input_ids.shape[1]:]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                self.respond(200, {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": args.served_model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": int(encoded.input_ids.shape[1]), "completion_tokens": int(tokens.shape[0]), "total_tokens": int(output.shape[1])}})
            except Exception as exc:
                self.respond(500, {"error": {"message": repr(exc)}})

    print(f"Serving {args.served_model_name} on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
