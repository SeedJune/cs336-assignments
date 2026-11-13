import modules
import torch
import json
from  bpe_tokenizer import Tokenizer

def sample(logits, temperature, top_p):
    probs = modules.Softmax(logits.float() / temperature, dim=-1)

    if top_p == 1:
        return torch.multinomial(probs, num_samples=1).item()

    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative_probs >= top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()
    sample_id = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids[sample_id].item()

@torch.inference_mode()
def generate(model:modules.Transformer_LM, tokenzier:Tokenizer, prompt:str, context_length=256, max_new_tokens=128, temperature=1.0, top_p=1.0):
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenzier.encode(prompt)
    eos_token = tokenzier.reverse_vocab[b"<|endoftext|>"]
    for i in range(max_new_tokens):
        inputs = torch.tensor(
            [tokens[-context_length:]],
            dtype=torch.long,
            device=device
        )
        logits = model(inputs)[0, -1, :]
        next_id = sample(logits, temperature, top_p)
        if next_id == eos_token:
            break
        tokens.append(next_id)

    return tokenzier.decode(tokens)


if __name__ == "__main__":
    
    config_path = "/home/jyc/workspace/llm-assignments/assignment1-basics/results/tinystories_20261112_022009/model_config.json"
    optimizer_path = "/home/jyc/workspace/llm-assignments/assignment1-basics/results/tinystories_20261112_022009/optimizer_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    with open(optimizer_path, "r", encoding="utf-8") as f:
        optimizer_config = json.load(f)
    
    tokenizer = Tokenizer.from_files("/home/jyc/workspace/llm-assignments/assignment1-basics/results/tinystories_bpe.pkl", special_tokens="<|endoftext|>")
    model = modules.Transformer_LM(**model_config).to("cuda:1")
    optimizer = modules.AdamW(model.parameters(), **optimizer_config)
    modules.load_checkpoint("/home/jyc/workspace/llm-assignments/assignment1-basics/results/tinystories_20261112_022009/checkpoint.pt", model, optimizer)
    
    prompt = "Of course, I can."
    results = generate(model,tokenzier=tokenizer, prompt=prompt)
    print(results)