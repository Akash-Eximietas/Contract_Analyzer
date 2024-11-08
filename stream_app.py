import asyncio
# Resolution for nemoguardrails to run async with streamlit application
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

import streamlit as st
from nemoguardrails import LLMRails, RailsConfig
import os

# Initialize the Nvidia API key (assuming environment variable setup)
if not os.environ.get("NVIDIA_API_KEY", ""):
    nvapi_key = "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"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

# Load your LLMRails configuration
config = RailsConfig.from_path("./config_nim")
rails = LLMRails(config)


def get_base64(bin_file):
    import base64
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file, background_opacity=1):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        opacity: %s; /* Adjust the background opacity level (0 to 1) */
    }
    </style>
    ''' % (bin_str, background_opacity)
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to handle user input and generate a response
def generate_response(user_input):
    try:
        response = rails.generate(prompt=user_input)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# Function to handle file upload and save files to the backend
def handle_file_upload(uploaded_files):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join('data', uploaded_file.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        return "Files uploaded successfully!"
    return "No files uploaded."

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.set_page_config(layout="wide")  # Set the layout to wide

# Add logo
st.image('Eximietas_Logo.png', use_container_width=False)
set_background('./back_20.png')
# Display app title and description
st.markdown("# Contract Analyzer Powered by NVIDIA NIM, Nemo Guardrails and LLAMA-INDEX")
# st.markdown("Uploaded files will be stored in the backend `data` folder.")

# File uploader
uploaded_files = st.file_uploader("Upload files", type=["pdf"], accept_multiple_files=True)

# File upload status
if st.button("Upload Files"):
    file_status = handle_file_upload(uploaded_files)
    st.write(file_status)

# Input box for text-based interaction
user_input = st.text_input("Got a Query?", "Type your message here...")
if st.button("Submit Query"):
    if user_input:
        output = generate_response(user_input)
        # Append the user input and output to chat history
        st.session_state.chat_history.append({"question": user_input, "answer": output})
        st.write("**Response:**", output)
    else:
        st.write("Please enter a query.")

# Display chat history
st.sidebar.markdown("## Conversation History")
st.sidebar.write(f"****")
for entry in st.session_state.chat_history:
    st.sidebar.write(f"**You:** {entry['question']}")
    st.sidebar.write(f"**Response:** {entry['answer']}")
    st.sidebar.write(f"****")