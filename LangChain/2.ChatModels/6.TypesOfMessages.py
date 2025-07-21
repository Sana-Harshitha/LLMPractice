from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_cohere import ChatCohere
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Initialize Chat Cohere model
llm = ChatCohere(
    cohere_api_key=api_key,
    model="command-r-plus"
)

# Initial messages list
messages = [
    SystemMessage(content="You are a friendly assistant.")
]

st.title("Chat Bot")

prompt = st.chat_input("Say something...")

if prompt:
    # Add user message
    messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        response = llm.invoke(messages)

    # Add AI response
    messages.append(AIMessage(content=response.content))
    st.chat_message("assistant").write(response.content)

print(messages)
