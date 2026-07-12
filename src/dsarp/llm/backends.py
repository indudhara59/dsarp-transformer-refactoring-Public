"""Disabled, mock, local-HF, and OpenAI-compatible LLM backends."""
from __future__ import annotations

import json,os
from abc import ABC,abstractmethod
from typing import Any


class LLMBackend(ABC):
    identifier:str
    @abstractmethod
    def generate(self,prompt:str,decoding:dict[str,Any])->str: ...


class DisabledBackend(LLMBackend):
    identifier="disabled"
    def generate(self,prompt:str,decoding:dict[str,Any])->str: raise RuntimeError("LLM validation is disabled")


class MockBackend(LLMBackend):
    identifier="mock"
    def __init__(self,response:dict[str,Any]): self.response=response; self.calls=0
    def generate(self,prompt:str,decoding:dict[str,Any])->str: self.calls+=1; return json.dumps(self.response)


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self,endpoint:str,model:str,api_key_env:str="DSARP_LLM_API_KEY",timeout:float=120): self.endpoint=endpoint.rstrip("/"); self.model=model; self.identifier=f"openai-compatible:{model}"; self.api_key_env=api_key_env; self.timeout=timeout
    def generate(self,prompt:str,decoding:dict[str,Any])->str:
        import httpx
        key=os.getenv(self.api_key_env); headers={"Authorization":f"Bearer {key}"} if key else {}
        payload={"model":self.model,"messages":[{"role":"user","content":prompt}],**decoding}; response=httpx.post(f"{self.endpoint}/v1/chat/completions",json=payload,headers=headers,timeout=self.timeout); response.raise_for_status(); return response.json()["choices"][0]["message"]["content"]


class LocalHFBackend(LLMBackend):
    def __init__(self,model_path:str,local_files_only:bool=True):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.identifier=f"local-hf:{model_path}"; self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=local_files_only); self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=local_files_only)
    def generate(self,prompt:str,decoding:dict[str,Any])->str:
        inputs=self.tokenizer(prompt,return_tensors="pt").to(self.model.device); output=self.model.generate(**inputs,max_new_tokens=int(decoding.get("max_tokens",1400)),temperature=float(decoding.get("temperature",0)),do_sample=float(decoding.get("temperature",0))>0); return self.tokenizer.decode(output[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)


def create_backend(config:dict[str,Any],mock_response:dict[str,Any]|None=None)->LLMBackend:
    kind=config.get("backend","disabled")
    if kind=="disabled": return DisabledBackend()
    if kind=="mock": return MockBackend(mock_response or {})
    if kind in {"vllm","openai_compatible","university"}: return OpenAICompatibleBackend(config["endpoint"],config["model"],config.get("api_key_env","DSARP_LLM_API_KEY"),config.get("timeout_seconds",120))
    if kind=="local_hf": return LocalHFBackend(config["model"])
    raise ValueError(f"Unsupported LLM backend: {kind}")

