import os
import time
import json
import uuid
import tempfile
from datetime import date
import streamlit as st
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

# 1. Custom CSS for Premium Tablet-Optimized Look & Feel
custom_css = """
<style>
    .reportview-container {
        font-family: 'Roboto', sans-serif;
    }
    /* Main Chat styling */
    .stChatFloatingInputContainer {
        border-top: 1px solid #dadce0;
        background-color: white;
    }
    /* Customize Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dadce0;
    }
    /* Custom button styles */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Persistent Multi-Turn Memory Management (Survives Hard Reloads)
if "session_id" not in st.query_params:
    session_id = str(uuid.uuid4())
    st.query_params["session_id"] = session_id
else:
    session_id = st.query_params["session_id"]

HISTORY_FILE = f"chat_history_{session_id}.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving history: {e}")

# Initialize Streamlit Session State
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = load_history()

# 3. Setup Multi-Key API Clients
raw_keys = input("Enter atleast one API key,plz separated by comma ").strip()
api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not api_keys:
    st.error("At least one valid Gemini API key must be provided.")
    st.stop()

# Cache clients so they don't re-initialize on every render
@st.cache_resource
def get_gemini_clients():
    return [genai.Client(api_key=key) for key in api_keys]

clients = get_gemini_clients()
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

# 4. DuckDuckGo Free Live Search
def fetch_free_web_context(user_query: str) -> str:
    time_keywords = ["today", "now", "yesterday", "news", "current", "who won", "latest", "weather", "2026", "2025", "price of"]
    if not any(word in user_query.lower() for word in time_keywords):
        return ""

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(user_query, max_results=3))
            if not results:
                return "\n[Web Search: No live results found.]\n"

            context = "\n[LIVE WEB SEARCH GROUNDING CONTEXT]\n"
            for r in results:
                context += f"• Source Title: {r['title']}\n  Snippet: {r['body']}\n\n"
            return context
    except Exception as e:
        return f"\n[Web Search: Temporary network error pulling live data: {e}]\n"

# 5. Application UI Header
st.title("✦ Google AI Mode")
st.caption("Multi-Key Failover Portal (Free Live Search & Persistent Memory Enabled)")
st.info("ℹ️ **Active Models:** `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash` | **Instant Key Switching**")

# Sidebar for auxiliary controls
with st.sidebar:
    st.header("Control Panel")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload File (Audio, Video, PDF, Images — Max: 2 GB)", 
        type=["mp3", "wav", "mp4", "pdf", "png", "jpg", "jpeg", "txt"]
    )
    
    # Reset Session Button
    if st.button("Reset Session Context", use_container_width=True):
        st.session_state.conversation_history = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

    # Diagnostics Info
    st.markdown("---")
