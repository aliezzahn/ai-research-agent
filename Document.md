# AI Agent Example: Line-by-Line Explanation

## A Detailed Guide to the Python Code and Its Outputs

This document explains the Python example files (`main.py` and `tools.py`) for building an AI research assistant using LangChain. It covers every line of code, its purpose, and all possible outputs and scenarios, including success cases, errors, and edge cases. By the end, you'll understand how the agent works and what to expect when running it.

---

## Table of Contents

1. [Overview](#chapter-1-overview)
2. [File Structure](#chapter-2-file-structure)
3. [Line-by-Line Explanation: `main.py`](#chapter-3-line-by-line-explanation-mainpy)
4. [Line-by-Line Explanation: `tools.py`](#chapter-4-line-by-line-explanation-toolspy)
5. [Sample Outputs and Scenarios](#chapter-5-sample-outputs-and-scenarios)
6. [Troubleshooting and Edge Cases](#chapter-6-troubleshooting-and-edge-cases)

---

## Chapter 1: Overview

The example creates an AI agent that acts as a research assistant. It:

- Uses a large language model (LLM) like Claude or GPT.
- Integrates tools (web search, Wikipedia, file saving).
- Structures output predictably using Pydantic.
- Accepts user queries and processes them with tools.

Key files:

- `main.py`: Core logic for the agent.
- `tools.py`: Tool definitions.
- `.env`: API key storage.
- `requirements.txt`: Dependencies.

---

## Chapter 2: File Structure

- **`main.py`**: Sets up the LLM, prompt, agent, and executor; handles user input and output parsing.
- **`tools.py`**: Defines three tools: DuckDuckGo search, Wikipedia query, and a custom save-to-file tool.
- **`.env`**: Stores API keys (e.g., `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).
- **`requirements.txt`**: Lists required packages (`langchain`, `pydantic`, etc.).

---

## Chapter 3: Line-by-Line Explanation: `main.py`

```python
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
```

- **`import os`**: Imports the `os` module to access environment variables.
- **`from dotenv import load_dotenv`**: Imports a function to load API keys from `.env`.
- **`from pydantic import BaseModel`**: Imports `BaseModel` to define structured output.
- **`from langchain_openai import ChatOpenAI`**: Imports OpenAI LLM integration.
- **`from langchain_anthropic import ChatAnthropic`**: Imports Anthropic (Claude) LLM integration.
- **`from langchain_core.prompts import ChatPromptTemplate`**: Imports a template for agent prompts.
- **`from langchain_core.output_parsers import PydanticOutputParser`**: Imports a parser for structured output.
- **`from langchain.agents import ...`**: Imports functions to create and run the agent.
- **`from tools import ...`**: Imports tools defined in `tools.py`.

```python
load_dotenv()
```

- Loads API keys from `.env` into the environment (e.g., `os.getenv("OPENAI_API_KEY")`).

```python
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

- Defines the LLM. Two options:
  - OpenAI (`ChatOpenAI`): Uses `gpt-4o-mini` model (commented out).
  - Anthropic (`ChatAnthropic`): Uses `claude-3-5-sonnet-20241022` model (active).
- `api_key=os.getenv(...)`: Fetches the key from `.env`.

```python
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
```

- Defines a Pydantic class for structured output with four fields:
  - `topic`: Research subject (string).
  - `summary`: Brief explanation (string).
  - `sources`: List of references (list of strings).
  - `tools_used`: Tools invoked (list of strings).

```python
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
```

- Creates a parser to convert LLM output into a `ResearchResponse` object.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant that will help generate a research paper. Answer the user query and use the necessary tools. Wrap the output in this format and provide no other text:\n{format_instructions}"),
    ("human", "{query}"),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())
```

- Defines a prompt template:
  - `system`: Instructs the agent on its role and output format.
  - `human`: Placeholder for the user’s query.
  - `chat_history` and `agent_scratchpad`: Placeholders for LangChain (auto-filled).
- `.partial(...)`: Inserts the Pydantic format instructions into the system message.

```python
tools = [search_tool, wiki_tool, save_tool]
```

- Lists the tools imported from `tools.py` for the agent to use.

```python
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
```

- Creates an agent combining the LLM, prompt, and tools.

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

- Sets up an executor to run the agent. `verbose=True` shows the agent’s thought process.

```python
if __name__ == "__main__":
    query = input("What can I help you research? ")
    raw_response = agent_executor.invoke({"query": query})
```

- Runs the script only if executed directly (not imported).
- Prompts the user for input and invokes the agent with the query.

```python
    try:
        structured_response = parser.parse(raw_response["output"][0]["text"])
        print("\nStructured Response:")
        print(structured_response)
    except Exception as e:
        print(f"\nError parsing response: {e}")
        print(f"Raw response: {raw_response}")
```

- Tries to parse the raw LLM output into a `ResearchResponse` object.
- Success: Prints the structured response.
- Failure: Prints the error and raw response for debugging.

---

## Chapter 4: Line-by-Line Explanation: `tools.py`

```python
# tools.py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from datetime import datetime
```

- Imports tools and utilities from LangChain and Python’s `datetime`.

```python
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information"
)
```

- Creates a web search tool using DuckDuckGo:
  - `search`: Instance of the search runner.
  - `search_tool`: Wraps it as a LangChain tool with a name, function, and description.

```python
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
```

- Sets up a Wikipedia tool:
  - `api_wrapper`: Limits results to 1 and content to 100 characters.
  - `wiki_tool`: Creates a query runner with the wrapper.

```python
def save_to_txt(data: str, filename: str = "research_output.txt"):
    with open(filename, "w") as f:
        f.write(f"Research Output\nTimestamp: {datetime.now()}\n\n{data}")
    return "File saved successfully"
```

- Defines a custom function to save data to a file:
  - `data`: The content to save (string).
  - `filename`: Output file (defaults to `research_output.txt`).
  - Writes a header, timestamp, and data; returns a success message.

```python
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file"
)
```

- Wraps `save_to_txt` as a tool with a name, function, and description.

---

## Chapter 5: Sample Outputs and Scenarios

### Scenario 1: Simple Query ("Tell me about sharks")

**Input**: `Tell me about sharks`
**Terminal Output**:

```bash
What can I help you research? Tell me about sharks

> Entering new AgentExecutor chain...
Invoking tool: search with query "shark biology habitat behavior research"
> Finished chain.

Structured Response:
topic='Sharks'
summary='Sharks are a group of elasmobranch fish characterized by a cartilaginous skeleton, five to seven gill slits, and pectoral fins not fused to the head.'
sources=['https://en.wikipedia.org/wiki/Shark']
tools_used=['search']
```

- **Explanation**: The agent uses the `search` tool, retrieves info from the web, and structures it.

### Scenario 2: Multi-Tool Query ("Hammerhead sharks")

**Input**: `Hammerhead sharks`
**Terminal Output**:

```bash
What can I help you research? Hammerhead sharks

> Entering new AgentExecutor chain...
Invoking tool: wiki_tool with query "Hammerhead shark"
Invoking tool: search with query "hammerhead shark research latest findings"
> Finished chain.

Structured Response:
topic='Hammerhead Sharks'
summary='Hammerhead sharks are known for their distinctive hammer-shaped heads, which enhance their sensory capabilities.'
sources=['Wikipedia: Hammerhead shark', 'https://www.sharkresearch.org']
tools_used=['wiki_tool', 'search']
```

- **Explanation**: The agent uses both `wiki_tool` and `search` for a richer response.

### Scenario 3: Save to File ("South East Asia population save to a file")

**Input**: `South East Asia population save to a file`
**Terminal Output**:

```bash
What can I help you research? South East Asia population save to a file

> Entering new AgentExecutor chain...
Invoking tool: wiki_tool with query "Southeast Asia"
Invoking tool: save_text_to_file with data...
> Finished chain.

Structured Response:
topic='South East Asia Population'
summary='Southeast Asia has a population of over 650 million, with a relatively young demographic.'
sources=['Wikipedia: Southeast Asia']
tools_used=['wiki_tool', 'save_text_to_file']
```

**File Output (`research_output.txt`)**:

```bash
Research Output
Timestamp: 2025-03-15 12:34:56.789012

topic='South East Asia Population'
summary='Southeast Asia has a population of over 650 million, with a relatively young demographic.'
sources=['Wikipedia: Southeast Asia']
tools_used=['wiki_tool', 'save_text_to_file']
```

- **Explanation**: The agent uses `wiki_tool` for data and `save_text_to_file` to save it.

---

## Chapter 6: Troubleshooting and Edge Cases

### 1. Missing API Key

**Scenario**: `.env` lacks a valid key.
**Output**:

```bash
Error: Missing API key for Anthropic/OpenAI
```

**Fix**: Add a valid key to `.env` and ensure the correct LLM is uncommented.

### 2. Parsing Error

**Scenario**: LLM outputs malformed data (e.g., missing fields).
**Output**:

```bash
Error parsing response: ValidationError: 1 validation error for ResearchResponse
summary: field required
Raw response: {'output': [{'text': '{"topic": "Sharks", "sources": ["Wikipedia"], "tools_used": ["search"]}'}]}
```

**Fix**: Adjust the prompt to enforce all fields or handle missing data gracefully.

### 3. Rate Limiting

**Scenario**: Too many requests to DuckDuckGo or Wikipedia.
**Output**:

```bash
> Entering new AgentExecutor chain...
Error: Rate limit exceeded for DuckDuckGo search
> Finished chain.
```

**Fix**: Wait, reduce tool usage, or use an API key-based service.

### 4. No Tools Used

**Scenario**: Query doesn’t trigger tools (e.g., "What is 2+2?").
**Output**:

```bash
Structured Response:
topic='2+2'
summary='2+2 equals 4.'
sources=[]
tools_used=[]
```

**Explanation**: The LLM answers directly without tools.

### 5. Invalid Query

**Scenario**: Empty or nonsensical input (e.g., "").
**Output**:

```bash
Structured Response:
topic='Unknown'
summary='Please provide a valid query.'
sources=[]
tools_used=[]
```

**Explanation**: The LLM may handle it gracefully or return an error depending on the model.

---

This guide covers the code and its behavior comprehensively. If you need further clarification or additional scenarios, let me know!
