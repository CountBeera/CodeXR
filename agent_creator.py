# src/agent_creator.py (Corrected Version 3.0 - The Robust Fix)

import os
from langchain import hub
# --- MODIFIED: Import the correct, modern agent constructor ---
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage

# Import our custom tool
from web_search_tool import get_web_search_tool

def create_agent(llm_model_name: str, memory: ConversationBufferWindowMemory):
    """
    Creates the main agent executor using the modern 'Tool Calling' method.
    This is more robust for tools that require structured inputs.
    """
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=llm_model_name,
        temperature=0.2
    )
    
    # Get the tool. Its default name is 'tavily_search_results_json'
    web_search_tool = get_web_search_tool()
    tools = [web_search_tool]
    
    # --- MODIFIED: Pull the prompt that is compatible with tool-calling agents ---
    prompt_template = hub.pull("hwchase17/openai-tools-agent")
    
    # Add a simpler system message. The tool-calling prompt is already very effective.
    system_message = "You are a helpful AI assistant."
    if not any(isinstance(msg, SystemMessage) for msg in memory.chat_memory.messages):
         memory.chat_memory.add_message(SystemMessage(content=system_message))

    # --- MODIFIED: Use `create_tool_calling_agent` ---
    # This constructor works seamlessly with the prompt and structured tools.
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    
    # Create the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        # This is less critical for tool-calling agents but still good practice
        handle_parsing_errors=True, 
        max_iterations=5,
        return_intermediate_steps=True 
    )
    
    return agent_executor