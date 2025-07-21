import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_to_token_ids(text, tokenizer):
    if isinstance(text, torch.Tensor):
        raise TypeError("Expected `text` to be a string, but got a tensor.")
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)  # shape: [1, T]


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    assert isinstance(top_k, (int, type(None))), "`top_k` must be an int or None"

    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        if logits.size(1) == 0:
            raise ValueError("Model returned empty logits. Likely due to empty input.")

        logits = logits[:, -1, :]  # Get logits of last token → shape [1, vocab]

        vocab_size = logits.shape[-1]

        if top_k is not None:
            top_k = int(top_k)
            if not (0 < top_k <= vocab_size):
                raise ValueError(f"Top-k must be > 0 and <= vocab size. Got top_k={top_k}, vocab_size={vocab_size}")
            
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
