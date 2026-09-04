import os
import time
import tempfile
from datetime import date
import streamlit as st
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

# 1. Safe & Modern Custom CSS (Won't crash the frontend)
custom_css = """
<style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Roboto', sans-serif;
    }
    /* Main Chat input styling */
    div[data-testid="stChatInput"] {
        border-top: 1px solid #dadce0;
    }
    /* Customize Sidebar background */
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

raw_keys = ""
# 2. Setup Multi-Key API Clients (Securely loaded from Streamlit Secrets)
if "GEMINI_API_KEYS" in st.secrets:
    raw_keys = st.secrets["GEMINI_API_KEYS"]


api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not api_keys:
    st.error("🔑 Error: No Gemini API keys found. Please add GEMINI_API_KEYS to your Streamlit secrets.")
    st.stop()

# Initialize Streamlit Session State for Chat History (Safe for Cloud)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Cache clients so they don't re-initialize on every user interaction
@st.cache_resource
def get_gemini_clients(keys_tuple):
    return [genai.Client(api_key=key) for key in keys_tuple]

# We pass a tuple to cache_resource because lists are unhashable
clients = get_gemini_clients(tuple(api_keys))
MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# 3. DuckDuckGo Free Live Search Grounding
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

# 4. Application UI Header
st.title("✦ Google AI Mode")
st.caption("Multi-Key Failover Portal (Free Live Search & Cloud Persistent Memory Enabled)")
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
        st.rerun()

    # Diagnostics Info
    st.markdown("---")
    st.subheader("System Diagnostics")
    st.write(f"🔑 Loaded API Keys: `{len(api_keys)}`")

# 5. Render Current Chat History
for turn in st.session_state.conversation_history:
    role = "user" if turn["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(turn["text"])

# 6. Accept User Input
if user_prompt := st.chat_input("Ask a question or request assistance..."):
    
    # Add User Message UI immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Rebuild context memory structure for API
    contents = []
    for turn in st.session_state.conversation_history:
        contents.append(f"{turn['role']}: {turn['text']}")
    contents.append(f"user: {user_prompt}")

    # Process File if Present
    temp_file_path = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            st.error("⚠️ File Limit Exceeded: File size exceeds maximum allowed of 2 GB.")
            st.stop()
            
        # Save payload to a temporary file
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
        try:
            os.remove(temp_file_path)
        except Exception:
            pass

    # 7. Save interactions into Streamlit Session State
    if success:
        st.session_state.conversation_history.append({'role': 'user', 'text': user_prompt})
        st.session_state.conversation_history.append({'role': 'model', 'text': full_response})
