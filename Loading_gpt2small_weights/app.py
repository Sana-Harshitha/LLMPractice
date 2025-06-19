import torch
import tiktoken
import streamlit as st
from LLM_core.GPTModel import GPTModel
from Generate_Next_Token import generate

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

@st.cache_resource
def load_model():
    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def text_to_token_ids(text):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0).to(device)

def token_ids_to_text(token_ids):
    return tokenizer.decode(token_ids.squeeze(0).tolist())

model = load_model()

# UI
st.title("Custom GPT-2 Text Generator")
prompt = st.text_input("Enter your prompt:", "Every effort moves you")
max_tokens = st.slider("Max new tokens", 1, 100, 25)
top_k = st.slider("Top-k", 10, 100, 50)
temperature = st.slider("Temperature", 0.5, 2.0, 1.0)

if st.button("Generate"):
    with st.spinner("Generating..."):
        input_ids = text_to_token_ids(prompt)
        if input_ids.shape[1] == 0:
             st.error("Prompt is empty or could not be tokenized. Please try a different input.")
        else:
            output_ids = generate(model, input_ids, max_tokens, 1024, temperature,top_k)

        output_text = token_ids_to_text(output_ids)
        st.text_area("Generated Output", output_text, height=None)
