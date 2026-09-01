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
st.caption("Active Models: `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` | Instant Key & Model Failover")

# Target models to cycle through instantly upon any error/quota hit
MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB limit

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load default keys from Streamlit Secrets if configured, otherwise leave blank
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
        help="Supports Audio (MP3), Video (MP4), PDFs, Images, and Text files."
    )
    
    if st.button("Reset Chat Context", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Session State Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main Input Bar
user_prompt = st.chat_input("Ask a question or request assistance...")

if user_prompt or uploaded_file:
    # Parse API Keys
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    
    if not api_keys:
        st.error("⚠️ Please enter at least one Gemini API Key in the sidebar.")
    else:
        prompt_text = user_prompt if user_prompt else "[Uploaded File]"
        
        # Render and save user message
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Build prior conversation history for Gemini context
        contents = []
        for msg in st.session_state.messages[:-1]:
            role_label = "user" if msg["role"] == "user" else "model"
            contents.append(f"{role_label}: {msg['content']}")
        contents.append(f"user: {prompt_text}")

        # System instructions with temporal context
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

                current_contents = list(contents)
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        sdk_file = client.files.upload(file=temp_file_path)
                        current_contents.append(sdk_file)
                    except Exception:
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
                    except Exception:
                        continue

            # Cleanup local temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if not success:
                full_response = "Execution Error: All API keys and model options were exhausted."
                response_placeholder.error(full_response)

            # Record final assistant response in session memory
            st.session_state.messages.append({"role": "assistant", "content": full_response})
