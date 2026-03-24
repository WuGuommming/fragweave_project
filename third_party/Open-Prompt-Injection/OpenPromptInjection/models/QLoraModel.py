import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .Model import Model


class QLoraModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])  # 128
        self.device = config["params"]["device"]

        self.base_model_id = config["model_info"]['name']
        self.ft_path = config["params"]['ft_path']

        # The original implementation always forced bitsandbytes 4-bit loading.
        # That path currently breaks in this environment because Transformers'
        # bnb replacement path expects nn.Module.set_submodule().
        # Default to standard loading, but allow opt-in 4-bit via config if needed.
        self.use_4bit = bool(config.get("params", {}).get("use_4bit", False))

        if "eval_only" not in config or not config["eval_only"]:
            self.base_model = self._load_base_model()

            if 'phi2' in self.provider or 'phi-2' in self.base_model_id:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_id,
                    add_bos_token=True,
                    trust_remote_code=True,
                    use_fast=False
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_id,
                    add_bos_token=True,
                    trust_remote_code=True
                )

            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.ft_path == '' or self.ft_path == 'base':
                self.ft_model = self.base_model
            else:
                try:
                    self.ft_model = PeftModel.from_pretrained(self.base_model, self.ft_path)
                except ValueError:
                    raise ValueError(f"Bad ft path: {self.ft_path}")
        else:
            self.ft_model = self.base_model = self.tokenizer = None

    def _preferred_dtype(self):
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _load_base_model(self):
        load_kwargs = {
            "trust_remote_code": True,
        }

        # Use explicit full-precision / mixed-precision loading by default.
        # This avoids the bitsandbytes 4-bit code path that is failing here.
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = self._preferred_dtype()
            if str(self.device).lower() == "auto":
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        # Optional best-effort 4-bit path for environments that explicitly request it.
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self._preferred_dtype(),
                )
                bnb_kwargs = dict(load_kwargs)
                bnb_kwargs["quantization_config"] = bnb_config
                bnb_kwargs["device_map"] = "auto"
                return AutoModelForCausalLM.from_pretrained(self.base_model_id, **bnb_kwargs)
            except Exception as e:
                print(f"[QLoraModel] 4-bit load failed, falling back to standard loading: {e}")

        model = AutoModelForCausalLM.from_pretrained(self.base_model_id, **load_kwargs)

        # If the model was not loaded with device_map, move it to the requested device.
        if "device_map" not in load_kwargs and torch.cuda.is_available():
            target_device = self.device
            if not isinstance(target_device, str) or target_device.lower() == "auto":
                target_device = "cuda"
            model = model.to(target_device)

        return model

    def _get_input_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| FT Path: {self.ft_path}\n{'-'*len(f'| Model name: {self.name}')}")

    def formatting_func(self, example):
        if isinstance(example, dict):
            input_split = example['input'].split('\nText: ')
        elif isinstance(example, str):
            input_split = example.split('\nText: ')
        else:
            raise ValueError(f'{type(example)} is not supported for querying Mistral')
        assert (len(input_split) == 2)
        text = f"### Instruction: {input_split[0]}\n### Text: {input_split[1]}"
        return text

    def query(self, msg):
        if self.ft_path == '' and 'DGDSGNH' not in msg:
            print('self.ft_model is None. Forward the query to the backend LLM')
            return self.backend_query(msg)

        processed_eval_prompt = self.formatting_func(msg)
        processed_eval_prompt = f'{processed_eval_prompt}\n### Response: '

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to(self._get_input_device(self.ft_model))

        self.ft_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.ft_model.generate(
                    **input_ids,
                    max_new_tokens=10,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )[0],
                skip_special_tokens=True
            ).replace(processed_eval_prompt, '')
        return output

    def query_localization(self, msg):
        if self.ft_path == '' and 'DGDSGNH' not in msg:
            print('self.ft_model is None. Forward the query to the backend LLM')
            return self.backend_query(msg)

        processed_eval_prompt = self.formatting_func(msg)
        processed_eval_prompt = f'{processed_eval_prompt}\n'

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to(self._get_input_device(self.ft_model))

        self.ft_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.ft_model.generate(
                    **input_ids,
                    max_new_tokens=10,
                    repetition_penalty=1.2,
                    do_sample=False,
                    temperature=0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )[0],
                skip_special_tokens=True
            ).replace(processed_eval_prompt, '')
        return output

    def backend_query(self, msg):
        if '\nText: ' in msg or (isinstance(msg, dict) and '\nText: ' in msg['input']):
            if isinstance(msg, dict):
                input_split = msg['input'].split('\nText: ')
            elif isinstance(msg, str):
                input_split = msg.split('\nText: ')
            else:
                raise ValueError(f'{type(msg)} is not supported for querying Mistral')
            assert (len(input_split) == 2)

            processed_eval_prompt = f"{input_split[0]}\nText: {input_split[1]}.{self.tokenizer.eos_token}"

        else:
            processed_eval_prompt = f"{msg} {self.tokenizer.eos_token}"

        input_ids = self.tokenizer(processed_eval_prompt, return_tensors="pt").to(self._get_input_device(self.base_model))

        self.base_model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.base_model.generate(
                    **input_ids,
                    max_new_tokens=self.max_output_tokens,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )[0],
                skip_special_tokens=True
            ).replace(processed_eval_prompt, '')
        return output
