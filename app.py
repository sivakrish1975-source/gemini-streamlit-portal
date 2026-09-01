import os
import streamlit as st
from google import genai
from google.genai import types

# Page Setup & Theme
st.set_page_config(
    page_title="Gemini Multi-Key Portal",
    page_icon="✦",
    layout="wide"
)

st.title("✦ Google AI Mode — Multi-Key Failover Portal")
st.caption("Active Models: `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` | Multi-Turn Context Memory Enabled")

# Target models to cycle through instantly upon any error/quota hit
MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB limit

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
    
    if st.button("Reset Chat Context", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Session State Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History from Session Memory
for message in st.session_state.messages:
    display_role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(display_role):
        text_content = ""
        for part in message.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                text_content += part["text"]
        st.markdown(text_content)

# Main Input Bar
user_prompt = st.chat_input("Ask a question or request assistance...")

if user_prompt:
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    
    if not api_keys:
        st.error("⚠️ Please enter at least one Gemini API Key in the sidebar.")
    else:
        # 1. Format user message in Gemini's native multi-turn dictionary structure
        user_turn = {
            "role": "user",
            "parts": [{"text": user_prompt}]
        }
        st.session_state.messages.append(user_turn)

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. System instructions
        system_config = types.GenerateContentConfig(
            system_instruction="Today's date is Tuesday, September 1, 2026. You are a helpful AI assistant built by Google."
        )

        clients = [genai.Client(api_key=key) for key in api_keys]

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
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

            # Failover Loop across (API Key x Model) matrix
            for client in clients:
                if success:
                    break

                # Clone past conversation history
                current_contents = [dict(msg) for msg in st.session_state.messages]

                # Attach file to the latest user turn if a file was uploaded
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        sdk_file = client.files.upload(file=temp_file_path)
                        current_contents[-1] = {
                            "role": "user",
                            "parts": [sdk_file, {"text": user_prompt}]
                        }
                    except Exception as e:
                        print(f"File upload failover error: {e}")
                        continue

                for model in MODELS_TO_TRY:
                    try:
                        stream = client.models.generate_content_stream(
                            model=model,
                            contents=current_contents,
                            config=system_config
                        )

                        full_response = ""
                        for chunk in stream:
                            if chunk.text:
                                full_response += chunk.text
                                response_placeholder.markdown(full_response + "▌")

                        response_placeholder.markdown(full_response)
                        success = True
                        break
                    except Exception as e:
                        print(f"Debug - Error with model {model}: {e}")
                        continue

            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if not success:
                full_response = "Execution Error: All API keys and model options were exhausted."
                response_placeholder.error(full_response)

            # 3. Store model response in session state memory for future turns
            st.session_state.messages.append({
                "role": "model",
                "parts": [{"text": full_response}]
            })
