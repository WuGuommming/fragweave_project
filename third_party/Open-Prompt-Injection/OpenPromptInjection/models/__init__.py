def create_model(config):
    """Factory method to create a LLM instance."""
    provider = config["model_info"]["provider"].lower()
    model_name = config["model_info"]["name"].lower()

    if provider == "palm2":
        from .PaLM2 import PaLM2
        model = PaLM2(config)
    elif provider == "vicuna":
        from .Vicuna import Vicuna
        model = Vicuna(config)
    elif provider == "gpt":
        from .GPT import GPT
        model = GPT(config)
    elif provider == "gpt_azure":
        from .GPT import GPTAzure
        model = GPTAzure(config)
    elif provider == "llama":
        from .Llama import Llama
        model = Llama(config)
    elif provider == "flan":
        from .Flan import Flan
        model = Flan(config)
    elif provider == "internlm":
        from .Internlm import Internlm
        model = Internlm(config)
    elif provider == "deepseek":
        if "distill" in model_name.lower():
            if "llama" in model_name.lower():
                from .DeepSeek import DeepSeekR1DistillLLaMAWrapper
                model = DeepSeekR1DistillLLaMAWrapper(config)
            elif "qwen" in model_name.lower():
                from .DeepSeek import DeepSeekR1DistillQwenWrapper
                model = DeepSeekR1DistillQwenWrapper(config)
            else:
                raise ValueError()
        else:
            from .DeepSeek import DeepSeekWrapper
            model = DeepSeekWrapper(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")

    return model


def create_qlora_model(config):
    """Factory method to create a QLoRA LLM."""
    provider = config["model_info"]["provider"].lower()

    if provider == "mistral":
        from .QLoraModel import QLoraModel
        model = QLoraModel(config)
    elif provider == "llama":
        from .QLoraModel import QLoraModel
        model = QLoraModel(config)
    elif provider == "llama3":
        from .Llama3 import Llama3
        model = Llama3(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")

    return model