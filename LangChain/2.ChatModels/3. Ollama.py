# Import the new Ollama integration
from langchain_ollama import ChatOllama

# Initialize the model
llm = ChatOllama(model="llama3")

# Create your prompt
prompt = "Hello! What can you do?"

# Get a response
response = llm.invoke(prompt)

# Print the response
print(response.content)
