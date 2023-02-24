import torch
import transformers


def is_codex_model(model_name):
    return model_name.startswith('code-davinci')


def get_model_by_name(model_name, device):
    if is_codex_model(model_name):
        # codex model
        from transformer_utils.codex import CodexModel, CodexTokenizer
        model = CodexModel(model_name)
        tokenizer = CodexTokenizer(model)

        return model, tokenizer
    else:
        # apps model
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

        model.to(device)

        if device == torch.device('cuda') and hasattr(model, 'parallelize'):
            model.parallelize()

        return model, tokenizer
