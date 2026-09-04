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
raw_keys = input("Enter your API key with comma seperating").strip()
api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not api_keys:
    st.error("At least one valid Gemini API key must be provided.")
    st.stop()

# Cache clients so they don't re-initialize on every render
@st.cache_resource
def get_gemini_clients():
    return [genai.Client(api_key=key) for key in api_keys]

clients = get_gemini_clients()
MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
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
                context += f"• Source Title: {r['title']}\n  Snippet: {r['body']}\n\n"
            return context
    except Exception as e:
        return f"\n[Web Search: Temporary network error pulling live data: {e}]\n"

# 5. Application UI Header
st.title("✦ Google AI Mode")
st.caption("Multi-Key Failover Portal (Free Live Search & Persistent Memory Enabled)")
st.info("ℹ️ **Active Models:** `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` | **Instant Key Switching**")

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
    st.subheader("System Diagnostics")
    st.write(f"🔑 Loaded API Keys: `{len(api_keys)}`")
    st.write(f"📂 Session Cache: `chat_history_{session_id[:8]}...json`")

# 6. Render Current Chat History
for turn in st.session_state.conversation_history:
    role = "user" if turn["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(turn["text"])

# 7. Accept User Input
if user_prompt := st.chat_input("Ask a question or request assistance..."):
    
    # Add User Message UI immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Rebuild context memory structure
    contents = []
    for turn in st.session_state.conversation_history:
        # Map back to API friendly turn strings
        contents.append(f"{turn['role']}: {turn['text']}")
    
    contents.append(f"user: {user_prompt}")

    # Process File if Present
    temp_file_path = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            st.error("⚠️ File Limit Exceeded: File size exceeds maximum allowed of 2 GB.")
            st.stop()
        
        # Save payload to a temporary file for the Files API to ingest
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

    # Check for search context
    web_context = fetch_free_web_context(user_prompt)
    system_config = types.GenerateContentConfig(
        system_instruction=(
            f"Today's date is: {date.today().strftime('%A, %B %d, %Y')} "
            "You are a helpful AI assistant built by Google. "
            f"Use the following real-time web context to help accurately answer the user's prompt:\n{web_context}"
        )
    )

    success = False
    start_time = time.perf_counter()

    # Add Streaming Container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Instant failover matrix across Clients x Models
        for client in clients:
            if success:
                break

            current_contents = list(contents)
            uploaded_file_ref = None

            if temp_file_path:
                try:
                    uploaded_file_ref = client.files.upload(file=temp_file_path)
                    current_contents.append(uploaded_file_ref)
                except Exception as upload_err:
                    print(f"Upload failed for client: {upload_err}. Moving to next client.")
                    continue

            for model_name in MODELS_TO_TRY:
                try:
                    response_stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=current_contents,
                        config=system_config
                    )
                    
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Final update to lock the UI string
                    message_placeholder.markdown(full_response)
                    success = True
                    latency = time.perf_counter() - start_time
                    st.toast(f"✅ Success via {model_name} in {latency:.2f}s", icon="🚀")
                    break
                except Exception as model_err:
                    print(f"Model {model_name} failed: {model_err}")
                    continue

        if not success:
            full_response = "❌ Execution Error: All API keys and model options were exhausted."
            message_placeholder.markdown(full_response)

    # Clean up temp files safely
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    # 8. Append interactions into user's persistent file session
    if success:
        st.session_state.conversation_history.append({'role': 'user', 'text': user_prompt})
        st.session_state.conversation_history.append({'role': 'model', 'text': full_response})
        save_history(st.session_state.conversation_history)
