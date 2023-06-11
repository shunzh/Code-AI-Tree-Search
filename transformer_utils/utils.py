import torch
import transformers


def get_model_by_name(model_name, device):
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

    model.to(device)

    if device == torch.device('cuda') and hasattr(model, 'parallelize'):
        model.parallelize()

    return model, tokenizer
