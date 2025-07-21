import os
from dotenv import load_dotenv  
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()   
# Get the Cohere API key from environment variables
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere LLM
llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-r-plus")

# Create the template
prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write an ad for {product} targeting {audience}."
)

formatted_prompt = prompt.format(
    product="a new smartwatch",
    audience="young professionals"
)

print("User: ",formatted_prompt)

# Generate response
response = llm.invoke(formatted_prompt)

print("\nModel response:")
print("AI: ",response.content)