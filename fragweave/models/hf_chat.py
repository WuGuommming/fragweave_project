from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fragweave.config import ModelConfig


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Global cache to avoid loading the same HF model/tokenizer multiple times.
# Keyed by (name_or_path, dtype_str, device_map).
_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


@dataclass
class HFChat:
    name: str
    tokenizer: any
    model: any
    max_new_tokens: int
    temperature: float
    top_p: float

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "HFChat":
        dtype_str = str(cfg.dtype)
        device_map = str(cfg.device_map)
        key = (cfg.name_or_path, dtype_str, device_map)

        if key in _MODEL_CACHE:
            tok, model = _MODEL_CACHE[key]
        else:
            dtype = None if cfg.dtype == "auto" else DTYPE_MAP.get(cfg.dtype)
            tok = AutoTokenizer.from_pretrained(cfg.name_or_path, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                cfg.name_or_path,
                torch_dtype=dtype,
                device_map=cfg.device_map,
                trust_remote_code=True,
            )
            model.eval()
            _MODEL_CACHE[key] = (tok, model)

        # Wrapper can still have different decoding params per "role"
        return cls(
            name=cfg.name_or_path,
            tokenizer=tok,
            model=model,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )

    def generate(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][input_ids.shape[-1]:]
        txt = self.tokenizer.decode(gen, skip_special_tokens=True)
        return txt.strip()
