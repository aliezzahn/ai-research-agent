# main.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load environment variables from .env file
load_dotenv()

# Set up the LLM (choose one based on your API key)
# Uncomment the one you want to use
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))

# Define the structured output format using Pydantic
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Create a parser to convert LLM output into the ResearchResponse format
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the prompt template with instructions for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant that will help generate a research paper. Answer the user query and use the necessary tools. Wrap the output in this format and provide no other text:\n{format_instructions}"),
    ("human", "{query}"),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

# List of tools the agent can use
tools = [search_tool, wiki_tool, save_tool]

# Create the agent with the LLM, prompt, and tools
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

# Set up the executor to run the agent (verbose=True shows the thought process)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Get user input and invoke the agent
if __name__ == "__main__":
    query = input("What can I help you research? ")
    raw_response = agent_executor.invoke({"query": query})

    # Parse the raw response into structured output
    try:
        structured_response = parser.parse(raw_response["output"][0]["text"])
        print("\nStructured Response:")
        print(structured_response)
    except Exception as e:
        print(f"\nError parsing response: {e}")
        print(f"Raw response: {raw_response}")