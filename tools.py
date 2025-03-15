# tools.py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from datetime import datetime

# Web search tool using DuckDuckGo
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information"
)

# Wikipedia tool with limited results and content
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Custom tool to save research output to a text file
def save_to_txt(data: str, filename: str = "research_output.txt"):
    with open(filename, "w") as f:
        f.write(f"Research Output\nTimestamp: {datetime.now()}\n\n{data}")
    return "File saved successfully"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file"
)