from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

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

        model_inputs = self.build_model_inputs(prompt=prompt, add_generation_prompt=True)
        input_ids = model_inputs["input_ids"]
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

    def build_model_inputs(self, prompt: str, add_generation_prompt: bool = False) -> Dict[str, Any]:
        """Tokenize a prompt with chat-template compatibility fallbacks across transformers versions."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = None
            attention_mask = None

            try:
                enc = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    return_dict=True,
                    return_tensors="pt",
                )
                if isinstance(enc, dict):
                    input_ids = enc.get("input_ids")
                    attention_mask = enc.get("attention_mask")
            except TypeError:
                enc = None

            if input_ids is None:
                enc = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    return_tensors="pt",
                )
                if isinstance(enc, torch.Tensor):
                    input_ids = enc
                elif isinstance(enc, dict) and "input_ids" in enc:
                    input_ids = enc["input_ids"]
                    attention_mask = enc.get("attention_mask")
                else:
                    raise RuntimeError("Unsupported output format from tokenizer.apply_chat_template")

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_text": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "decoded_text": self.tokenizer.decode(input_ids[0], skip_special_tokens=False),
        }

    def forward_for_analysis(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Any:
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=use_cache,
                return_dict=True,
            )

    def encode_prompt_states(
        self,
        prompt: str,
        *,
        add_generation_prompt: bool = True,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        model_inputs = self.build_model_inputs(prompt=prompt, add_generation_prompt=add_generation_prompt)
        outputs = self.forward_for_analysis(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            output_hidden_states=True,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Model forward did not return hidden states for analysis.")
        hidden_states = [h.detach().cpu() for h in outputs.hidden_states]
        out: Dict[str, Any] = {
            "input_ids": model_inputs["input_ids"].detach().cpu(),
            "attention_mask": model_inputs["attention_mask"].detach().cpu(),
            "hidden_states": hidden_states,
            "token_text": model_inputs["token_text"],
            "decoded_text": model_inputs["decoded_text"],
        }
        if output_attentions:
            if outputs.attentions is None:
                raise RuntimeError("Model forward did not return attentions for analysis.")
            out["attentions"] = [a.detach().cpu() for a in outputs.attentions]
        return out

    def score_target_text(
        self,
        prompt: str,
        target_text: str,
        *,
        add_generation_prompt: bool = True,
    ) -> Dict[str, Any]:
        model_inputs = self.build_model_inputs(prompt=prompt, add_generation_prompt=add_generation_prompt)
        prompt_ids = model_inputs["input_ids"]
        prompt_attention = model_inputs["attention_mask"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids
        if target_ids.numel() == 0:
            raise ValueError("target_text tokenized to zero tokens; cannot score empty target.")

        full_ids = torch.cat([prompt_ids, target_ids], dim=1).to(self.model.device)
        full_mask = torch.cat([prompt_attention, torch.ones_like(target_ids)], dim=1).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=full_ids, attention_mask=full_mask, return_dict=True)
            logits = outputs.logits[:, :-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        target_pos = torch.arange(target_ids.shape[1], device=full_ids.device) + prompt_ids.shape[1] - 1
        token_logprobs = log_probs[0, target_pos, full_ids[0, prompt_ids.shape[1]:]]
        return {
            "prompt_input_ids": prompt_ids.detach().cpu(),
            "target_input_ids": target_ids.detach().cpu(),
            "target_tokens": self.tokenizer.convert_ids_to_tokens(target_ids[0].tolist()),
            "token_logprobs": token_logprobs.detach().cpu(),
            "total_logprob": float(token_logprobs.sum().item()),
            "mean_logprob": float(token_logprobs.mean().item()),
        }

    def compute_input_attribution(
        self,
        prompt: str,
        target_text: str,
        *,
        method: str = "grad_x_input",
        add_generation_prompt: bool = True,
        ig_steps: int = 16,
    ) -> Dict[str, Any]:
        method_low = method.lower().strip()
        if method_low not in {"grad_x_input", "integrated_gradients"}:
            raise ValueError(f"Unsupported attribution method: {method}")

        model_inputs = self.build_model_inputs(prompt=prompt, add_generation_prompt=add_generation_prompt)
        prompt_ids = model_inputs["input_ids"].to(self.model.device)
        prompt_mask = model_inputs["attention_mask"].to(self.model.device)
        target_ids = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
        if target_ids.numel() == 0:
            raise ValueError("target_text tokenized to zero tokens; attribution requires non-empty target.")

        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(target_ids)], dim=1)
        prompt_len = prompt_ids.shape[1]

        embed_layer = self.model.get_input_embeddings()
        input_embeds = embed_layer(full_ids).detach()
        input_embeds.requires_grad_(True)

        def _score_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
            outputs = self.model(inputs_embeds=embeds, attention_mask=full_mask, return_dict=True)
            logits = outputs.logits[:, :-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            target_pos = torch.arange(target_ids.shape[1], device=log_probs.device) + prompt_len - 1
            token_scores = log_probs[0, target_pos, full_ids[0, prompt_len:]]
            return token_scores.sum()

        if method_low == "grad_x_input":
            score = _score_from_embeds(input_embeds)
            grad = torch.autograd.grad(score, input_embeds)[0]
            token_attr = (grad * input_embeds).sum(dim=-1)[0, :prompt_len]
        else:
            baseline = torch.zeros_like(input_embeds)
            alphas = torch.linspace(0.0, 1.0, steps=max(2, ig_steps), device=input_embeds.device)
            grad_acc = torch.zeros_like(input_embeds)
            for alpha in alphas:
                emb_step = baseline + alpha * (input_embeds - baseline)
                emb_step.requires_grad_(True)
                score = _score_from_embeds(emb_step)
                grad_step = torch.autograd.grad(score, emb_step)[0]
                grad_acc += grad_step
            avg_grad = grad_acc / float(alphas.shape[0])
            token_attr = ((input_embeds - baseline) * avg_grad).sum(dim=-1)[0, :prompt_len]

        return {
            "input_ids": prompt_ids.detach().cpu(),
            "token_text": model_inputs["token_text"],
            "decoded_text": model_inputs["decoded_text"],
            "token_attribution": token_attr.detach().cpu(),
            "method": method_low,
            "target_text": target_text,
        }
