from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Define schema with Pydantic
class CompanyInfo(BaseModel):
    company_name: str = Field(..., description="The name of the company")
    headquarters: str = Field(..., description="Location of HQ")
    founded_year: int = Field(..., description="Year founded")

parser = JsonOutputParser(pydantic_object=CompanyInfo)

# Prompt
prompt = PromptTemplate(
    template="Extract the following information:\n{format_instructions}\n\nText: {input_text}",
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Groq model
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, max_tokens=512)

# LCEL chain
chain = prompt | llm | parser

text = "OpenAI is headquartered in San Francisco and was founded in 2015."
result = chain.invoke({"input_text": text})

print(result)
