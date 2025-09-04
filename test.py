# test_search.py

import os
from dotenv import load_dotenv

# Important: Load environment variables before importing from your src
load_dotenv()

# Make sure your TAVILY_API_KEY is loaded
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in .env file")

# Now import your tool function
from web_search_tool import get_web_search_tool

def run_test():
    """
    Tests the Tavily web search tool by invoking it with a sample query.
    """
    print("--- Testing Tavily Web Search Tool ---")
    
    # 1. Initialize the tool
    tavily_tool = get_web_search_tool()
    print(f"Tool Name: {tavily_tool.name}")
    print(f"Tool Description: {tavily_tool.description}\n")
    
    # 2. Define a query
    query = "What are the main new features in Apple's VisionOS 2?"
    print(f"Executing query: '{query}'\n")
    
    # 3. Invoke the tool
    # The .invoke() method runs the tool directly.
    results = tavily_tool.invoke({"query": query})
    
    # 4. Print the results
    print("--- Results ---")
    print(results)
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_test()