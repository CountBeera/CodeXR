# src/tools/web_search_tool.py

from langchain_community.tools.tavily_search import TavilySearchResults

def get_web_search_tool():
    """
    Initializes and returns the Tavily web search tool.
    We are not setting a custom 'name' here, so it will default to 
    'tavily_search_results_json', which the tool-calling agent understands natively.
    """
    search_tool = TavilySearchResults(
        max_results=3,
        description="A search engine useful for searching the internet for current events, recent information, or general knowledge that is not related to AR/VR development."
    )
    return search_tool