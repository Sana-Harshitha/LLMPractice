import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load .env
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Init Cohere LLM
llm = ChatCohere(
    cohere_api_key=api_key,
    model="command-r-plus"
)

# Streamlit page setup
st.set_page_config(page_title="LangChain ChatBot")
st.title("Chatbot")

# Initialize session state chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build ChatPromptTemplate with dynamic memory placeholder
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who answers questions clearly."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# Input field for user message
user_input = st.chat_input("Type your message...")

# Render previous chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# If user submits a new prompt
if user_input:
    # Add user message to history
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Format messages with current history
    messages = chat_prompt.format_messages(
        input=user_input,
        chat_history=st.session_state.chat_history[:-1]  # exclude latest user message
    )

    # Get response from LLM
    with st.spinner("Thinking..."):
        response = llm.invoke(messages)

    # Add assistant message to history
    st.chat_message("assistant").write(response.content)
    st.session_state.chat_history.append(AIMessage(content=response.content))
