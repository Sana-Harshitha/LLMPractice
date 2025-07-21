import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Initialize model
llm = ChatCohere(
    cohere_api_key=api_key,
    model="command-r-plus"
)

st.set_page_config(page_title="Dynamic Prompt Chatbot")
st.title("🤖 Dynamic Prompt Template Chatbot")

# User chooses specialty
specialty = st.selectbox(
    "Choose assistant specialty:",
    ["General Help", "Python Programming", "Travel Advice", "Business Strategy"]
)

# Define template
prompt_template = PromptTemplate(
    input_variables=["specialty"],
    template="You are a helpful assistant specialized in {specialty}. Answer the user's questions clearly."
)

# Create system message dynamically
system_prompt = prompt_template.format(specialty=specialty)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content=system_prompt)
    ]

# Rebuild system message on specialty change
if (
    st.session_state["messages"]
    and isinstance(st.session_state["messages"][0], SystemMessage)
):
    st.session_state["messages"][0] = SystemMessage(content=system_prompt)

# Display chat history
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
prompt = st.chat_input("Ask something...")

if prompt:
    # Add user message
    st.session_state["messages"].append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Invoke LLM
    with st.spinner("Thinking..."):
        response = llm.invoke(st.session_state["messages"])

    # Add assistant response
    st.session_state["messages"].append(AIMessage(content=response.content))
    st.chat_message("assistant").write(response.content)
