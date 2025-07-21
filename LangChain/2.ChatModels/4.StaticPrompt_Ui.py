import cohere
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
co = cohere.Client(api_key)

st.title("Simple Cohere Chatbot (NO history)")

prompt = st.chat_input("Type your message...")

if prompt:
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=300
        )

    # Cohere returns a generations list
    st.chat_message("assistant").write(response.generations[0].text.strip())
