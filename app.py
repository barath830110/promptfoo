from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="llama3.2", temperature=0)

prompt_template = PromptTemplate.from_template(
    "{input}"
)

chain = prompt_template | llm


def call_api(prompt, options=None, context=None):
    result = chain.invoke({"input": prompt})
    return {
        "output": result
    }
