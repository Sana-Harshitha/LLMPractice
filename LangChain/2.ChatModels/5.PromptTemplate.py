
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
# Initialize Cohere LLM
llm = ChatCohere(cohere_api_key="Wg5jgYLIffSDxJHnIeCkXNOI15GDQ2PlR4iKhAlg", model="command-r-plus")

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