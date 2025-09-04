# app.py (Refactored for UI only)

import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder

# --- NEW: Import our agent creator and new LangChain components ---
from agent_creator import create_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage

# --- Load Environment Variables ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# --- AESTHETIC & UI FIXES: ADVANCED CSS (Your existing CSS is perfect) ---
st.markdown("""
<style>
/* ... your existing CSS here ... */
</style>
""", unsafe_allow_html=True)


# --- API Key and Client Initialization ---
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY") # NEW Check

if not groq_api_key or not tavily_api_key:
    st.error("API keys for GROQ and TAVILY not found! Please set them in your .env file.", icon="🔥")
    st.stop()
whisper_client = Groq(api_key=groq_api_key)


# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    # Recommended models for tool use: gemma2, llama3
    LLM_MODEL = st.selectbox("Choose a model", ["openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct", "gemma2-9b-it"])
    WHISPER_MODEL = "whisper-large-v3"
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, LangChain, and Groq.")
    if st.button("Clear Chat History"):
        st.session_state.clear() # Clear all session state
        st.rerun()

# --- LangChain Callback Handler for Streaming UI ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
    # NEW: A simple way to show the user which tool is being used
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.container.markdown(f"**Thinking... (using `{serialized['name']}`)**")


# --- Initialize Agent and Memory in Session State ---
# This block now uses our new agent creator
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=10, return_messages=True, memory_key="chat_history"
    )

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_agent(
        llm_model_name=LLM_MODEL,
        memory=st.session_state.memory
    )

# --- UI Rendering ---
st.title("🤖 COdeXR: AI Agent for AR/VR Dev")

chat_container = st.container()
with chat_container:
    for message in st.session_state.memory.chat_memory.messages:
        # Don't display the custom system prompt we added in the agent
        if isinstance(message, SystemMessage) and "You are a helpful AI assistant" in message.content: continue
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# --- Core Logic Functions ---
def process_user_input(prompt):
    if not prompt or not prompt.strip(): return

    with chat_container:
        with st.chat_message("user"): st.markdown(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            response_container = st.empty()
            handler = StreamHandler(response_container)
            
            # --- MODIFIED: Use the agent_executor instead of conversation chain ---
            response = st.session_state.agent_executor.invoke(
                {"input": prompt},
                config={"callbacks": [handler]}
            )
            # The final answer is in the 'output' key of the response
            response_container.markdown(response['output'])

# (Your transcribe_audio and handle_submission functions are great, no changes needed)
def transcribe_audio(audio_bytes):
    with st.spinner("Transcribing audio... 🎤"):
        try:
            with open("temp_audio.wav", "wb") as f: f.write(audio_bytes)
            with open("temp_audio.wav", "rb") as audio_file:
                transcription = whisper_client.audio.transcriptions.create(
                    file=("temp_audio.wav", audio_file.read()), model=WHISPER_MODEL, response_format="text"
                )
            os.remove("temp_audio.wav")
            return transcription
        except Exception as e:
            st.error(f"Transcription Error: {e}", icon="🔥")
            return None

def handle_submission():
    input_value = st.session_state.text_input_value
    process_user_input(input_value)
    st.session_state.text_input_value = ""


# --- Sticky Input Footer (No changes needed here) ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
footer_cols = st.columns([10, 1])

with footer_cols[0]:
    st.text_input(
        "Ask me anything...",
        key="text_input_value",
        on_change=handle_submission,
        label_visibility="collapsed"
    )

with footer_cols[1]:
    audio_bytes = audio_recorder(text="", icon_size="2x")
    if audio_bytes:
        transcribed_text = transcribe_audio(audio_bytes)
        process_user_input(transcribed_text)

st.markdown('</div>', unsafe_allow_html=True)