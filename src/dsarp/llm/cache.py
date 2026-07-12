"""Content-addressed LLM cache and append-only audit logs."""
import hashlib,json
from pathlib import Path
from typing import Any


class LLMCache:
    def __init__(self,directory:Path): self.directory=directory; directory.mkdir(parents=True,exist_ok=True)
    @staticmethod
    def key(model:str,prompt_version:str,candidate_id:str,prompt:str,decoding:dict[str,Any])->str:
        payload=json.dumps({"model":model,"prompt_version":prompt_version,"candidate_id":candidate_id,"prompt":prompt,"decoding":decoding},sort_keys=True); return hashlib.sha256(payload.encode()).hexdigest()
    def get(self,key:str)->dict[str,Any]|None:
        path=self.directory/f"{key}.json"; return json.loads(path.read_text()) if path.exists() else None
    def put(self,key:str,value:dict[str,Any])->Path:
        path=self.directory/f"{key}.json"; path.write_text(json.dumps(value,indent=2,sort_keys=True)); return path


def append_jsonl(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as stream: stream.write(json.dumps(value,sort_keys=True,default=str)+"\n")

