# Building an AI Agent from Scratch in Python

## A Beginner's Guide to Creating a Research Assistant with LangChain

This guide walks you through building an AI agent from scratch using Python. We'll use popular frameworks like LangChain to integrate large language models (LLMs) such as Claude or GPT, give the agent access to tools like Wikipedia and web search, and structure its output for use in your code. By the end, you'll have a functional research assistant that can save its findings to a file. Let's dive in!

---

## Table of Contents

1. [Introduction](#chapter-1-introduction)
2. [Prerequisites](#chapter-2-prerequisites)
3. [Setting Up the Environment](#chapter-3-setting-up-the-environment)
4. [Writing the Core Code](#chapter-4-writing-the-core-code)
5. [Adding Tools](#chapter-5-adding-tools)
6. [Running the Agent](#chapter-6-running-the-agent)
7. [Sample Outputs](#chapter-7-sample-outputs)
8. [Conclusion](#chapter-8-conclusion)

---

## Chapter 1: Introduction

In this tutorial, you'll learn how to build an AI agent step-by-step in Python. The agent will act as a research assistant, capable of answering queries, using tools like Wikipedia and DuckDuckGo search, and saving results to a text file. We'll use LangChain to integrate LLMs and structure the output predictably.

### Demo of the Finished Project

The agent asks: "What can I help you research?" For example, if you input "Tell me about LangChain and its applications" and request it to save to a file, it:

- Searches Wikipedia and the web.
- Provides a structured response with a topic, summary, sources, and tools used.
- Saves the output to a text file with a timestamp.

---

## Chapter 2: Prerequisites

Before starting, ensure you have the following:

1. **Python**: Version 3.10 or higher recommended. Install from [python.org](https://www.python.org/).
2. **Code Editor**: Visual Studio Code (VS Code) is recommended.
3. **API Keys**: You'll need API keys for an LLM provider (e.g., OpenAI or Anthropic). Instructions provided later.

---

## Chapter 3: Setting Up the Environment

### Step 1: Create a Project Folder

- Open VS Code.
- Go to `File > Open Folder`.
- Create a new folder (e.g., `AI_Agent_Tutorial`) and open it.

### Step 2: Create a Requirements File

Create a file named `requirements.txt` with these dependencies:

```plaintext
langchain
langchain-openai
langchain-anthropic
wikipedia
duckduckgo-search
python-dotenv
pydantic
```

### Step 3: Set Up a Virtual Environment

1. Open a terminal in VS Code.
2. Run:
   - Windows: `python -m venv venv`
   - Mac/Linux: `python3 -m venv venv`
3. Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Create Additional Files

- `main.py`: Main logic for the agent.
- `tools.py`: Custom and external tools.
- `.env`: Environment variables for API keys.

---

## Chapter 4: Writing the Core Code

### Step 1: Set Up API Keys in `.env`

Create a `.env` file and add your API key(s):

```bash
# For OpenAI
OPENAI_API_KEY="your-openai-api-key"

# For Anthropic (Claude)
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

- **Get OpenAI Key**: Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys), generate a key, and paste it.
- **Get Anthropic Key**: Go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys), create a key, and paste it.

### Step 2: Write `main.py`

This file sets up the LLM, prompt, and agent.

```python
# main.py
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load environment variables
load_dotenv()

# Set up the LLM (choose one)
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")  # or ChatOpenAI(model="gpt-4o-mini")

# Define the output structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Create parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Set up prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant that will help generate a research paper. Answer the user query and use the necessary tools. Wrap the output in this format and provide no other text:\n{format_instructions}"),
    ("human", "{query}"),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

# Define tools
tools = [search_tool, wiki_tool, save_tool]

# Create agent
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Get user input and run agent
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

# Parse and display structured output
try:
    structured_response = parser.parse(raw_response["output"][0]["text"])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}")
    print(f"Raw response: {raw_response}")
```

### Explanation

- **LLM**: Configures Claude or GPT with an API key loaded from `.env`.
- **ResearchResponse**: Defines the structured output format using Pydantic.
- **Prompt**: Instructs the agent on its role and output format.
- **Agent**: Combines LLM, prompt, and tools.
- **Executor**: Runs the agent and shows its thought process (`verbose=True`).

---

## Chapter 5: Adding Tools

### Step 1: Write `tools.py`

This file defines tools for web search, Wikipedia, and saving to a file.

```python
# tools.py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from datetime import datetime

# Web search tool (DuckDuckGo)
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information"
)

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Custom save-to-file tool
def save_to_txt(data: str, filename: str = "research_output.txt"):
    with open(filename, "w") as f:
        f.write(f"Research Output\nTimestamp: {datetime.now()}\n\n{data}")

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file"
)
```

### Explanation

- **Search Tool**: Uses DuckDuckGo to search the web.
- **Wikipedia Tool**: Queries Wikipedia with a limit of 1 result and 100 characters.
- **Save Tool**: Custom function to save output to a text file with a timestamp.

---

## Chapter 6: Running the Agent

1. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
2. Run the script:
   ```bash
   python main.py
   ```
3. Enter a query, e.g., "South East Asia population save to a file".

---

## Chapter 7: Sample Outputs

### Example 1: Query - "Tell me about sharks"

**Terminal Output**:

```bash
> What can I help you research? Tell me about sharks
[AgentExecutor Chain]
Invoking tool: search with query "shark biology habitat behavior research"
[Output]
topic='Sharks'
summary='Sharks are a group of elasmobranch fish characterized by a cartilaginous skeleton, five to seven gill slits, and pectoral fins not fused to the head.'
sources=['https://en.wikipedia.org/wiki/Shark']
tools_used=['search']
```

### Example 2: Query - "Hammerhead sharks"

**Terminal Output**:

```bash
> What can I help you research? Hammerhead sharks
[AgentExecutor Chain]
Invoking tool: wiki_tool with query "Hammerhead shark"
Invoking tool: search with query "hammerhead shark research latest findings"
[Output]
topic='Hammerhead Sharks'
summary='Hammerhead sharks are known for their distinctive hammer-shaped heads, which enhance their sensory capabilities.'
sources=['Wikipedia: Hammerhead shark', 'https://www.sharkresearch.org']
tools_used=['wiki_tool', 'search']
```

### Example 3: Query - "South East Asia population save to a file"

**Terminal Output**:

```bash
> What can I help you research? South East Asia population save to a file
[AgentExecutor Chain]
Invoking tool: wiki_tool with query "Southeast Asia"
Invoking tool: save_text_to_file with data...
[Output]
topic='South East Asia Population'
summary='Southeast Asia has a population of over 650 million, with a relatively young demographic.'
sources=['Wikipedia: Southeast Asia']
tools_used=['wiki_tool', 'save_text_to_file']
```

**File Output (`research_output.txt`)**:

```bash
Research Output
Timestamp: 2025-03-15 10:00:00
topic='South East Asia Population'
summary='Southeast Asia has a population of over 650 million, with a relatively young demographic.'
sources=['Wikipedia: Southeast Asia']
tools_used=['wiki_tool', 'save_text_to_file']
```

---

## Chapter 8: Conclusion

Congratulations! You've built an AI research assistant from scratch using Python and LangChain. It can:

- Answer queries using LLMs like Claude or GPT.
- Use tools like Wikipedia, web search, and custom file-saving.
- Structure output predictably with Pydantic.

### Next Steps

- Add more tools (e.g., API calls, databases).
- Improve the prompt for better responses.
- Experiment with different LLMs or models.
