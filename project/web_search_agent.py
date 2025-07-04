"""
web_search_agent.py
A simple web search agent using DuckDuckGo via LangChain for fallback answers.
"""
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

def web_search(query):
    """
    Perform a web search using DuckDuckGoSearchRun from LangChain.
    Args:
        query (str): The user's question.
    Returns:
        str: The top web search result or a message if not available.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result if result else "No relevant web search results found."
    except Exception as e:
        return f"Web search error: {e}"
