import os
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Gemini Multi-Key Portal",
    page_icon="✦",
    layout="wide"
)

st.title("✦ Google AI Mode — Multi-Key Failover Portal")
st.caption("Active Models: `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` | Multi-Turn Memory Enabled")

MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB limit

# 1. PERSISTENT MEMORY (Replaces the global conversation_history list)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_keys = st.secrets.get("GEMINI_KEYS", "") if hasattr(st, "secrets") else ""
    
    raw_keys = st.text_input(
        "Gemini API Keys (Comma-separated):",
        value=default_keys,
        type="password",
        help="Paste one or more Gemini API keys separated by commas."
    )
    
    uploaded_file = st.file_uploader(
        "Upload File (Max 2GB)",
        type=None,
        help="Supports Audio, Video, PDFs, Images, and Text files."
    )
    
    if st.button("Reset Context", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

# Display Chat History from persistent session memory
for turn in st.session_state.conversation_history:
    role = "user" if turn["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(turn["text"])

# Main Chat Input
user_message = st.chat_input("Ask a question or request assistance...")

if user_message or uploaded_file:
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    
    if not api_keys:
        st.error("⚠️ Please enter at least one Gemini API Key in the sidebar.")
    else:
        prompt_text = user_message if user_message else "[Uploaded File]"
        
        # Render user message on UI
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # 2. EXACT MEMORY REBUILD LOGIC FROM YOUR WORKING COLAB CODE
        contents = []
        for turn in st.session_state.conversation_history:
            contents.append(f"{turn['role']}: {turn['text']}")
        contents.append(f"user: {prompt_text}")

        # System instructions with temporal context
        system_config = types.GenerateContentConfig(
            system_instruction="Today's date is Tuesday, September 1, 2026. You are a helpful AI assistant built by Google."
        )

        clients = [genai.Client(api_key=key) for key in api_keys]

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            success = False

            # Handle file upload if attached
            temp_file_path = None
            if uploaded_file is not None:
                if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                    st.error("⚠️ File Limit Exceeded: Selected file exceeds maximum allowed size of 2 GB.")
                    st.stop()
                
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # 3. FAILOVER LOOP (EXACT LOGIC FROM YOUR WORKING COLAB CODE)
            for client in clients:
                if success:
                    break

                current_contents = list(contents)
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        sdk_file = client.files.upload(file=temp_file_path)
                        current_contents.append(sdk_file)
                    except Exception:
                        continue

                for model_name in MODELS_TO_TRY:
                    try:
                        response_stream = client.models.generate_content_stream(
                            model=model_name,
                            contents=current_contents,
                            config=system_config
                        )

                        response_text = ""
                        for chunk in response_stream:
                            if chunk.text:
                                response_text += chunk.text
                                response_placeholder.markdown(response_text + "▌")

                        response_placeholder.markdown(response_text)
                        success = True
                        break
                    except Exception as e:
                        print(f"Debug - Key/Model ({model_name}) error: {e}")
                        continue

            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if not success:
                response_text = "Execution Error: All API keys and model options were exhausted."
                response_placeholder.error(response_text)

            # 4. SAVE TO PERSISTENT MEMORY (EXACT LOGIC FROM YOUR COLAB CODE)
            if success and response_text:
                st.session_state.conversation_history.append({'role': 'user', 'text': prompt_text})
                st.session_state.conversation_history.append({'role': 'model', 'text': response_text})
