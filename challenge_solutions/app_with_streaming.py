import streamlit as st  # Import Streamlit library for creating web interfaces
from langchain_community.llms.ollama import Ollama  # Import Ollama integration from LangChain
from langchain.callbacks.base import BaseCallbackHandler  # Import for streaming callback

# App title - this will appear at the top of the page
st.title("🤖 Chat with DeepSeek (Streaming)")

# Add a brief description
st.markdown("This app demonstrates streaming responses character-by-character")

# Create a dropdown to select from available Ollama models
try:
    # Try to get available models from Ollama's local API
    import requests
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        # Extract model names from the response
        available_models = [model["name"] for model in response.json()["models"]]
    else:
        # Fallback if API returns error but is available
        available_models = ["deepseek-coder", "deepseek-r1:1.5b"]
except Exception:
    # Fallback if Ollama isn't running or other error occurs
    st.warning("⚠️ Couldn't connect to Ollama API. Is Ollama running?")
    available_models = ["deepseek-coder", "deepseek-r1:1.5b"]

# Create a dropdown in the sidebar to select the model
with st.sidebar:
    st.header("Model Settings")
    selected_model = st.selectbox(
        "Select a model",
        options=available_models,
        index=0 if available_models else 0,
        help="Choose which Ollama model to chat with"
    )

# Create custom streaming handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Initialize chat history in session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user to type messages
prompt = st.chat_input("Ask me anything...")

# If the user sends a message
if prompt:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner while waiting
    with st.chat_message("assistant"):
        # Create an empty placeholder for the streaming response
        response_placeholder = st.empty()
        
        try:
            # Initialize the streaming handler
            stream_handler = StreamHandler(response_placeholder)
            
            # Initialize Ollama with streaming
            llm = Ollama(model=selected_model, callbacks=[stream_handler])
            
            # Generate the full prompt with history context
            full_prompt = "You are a helpful assistant. Keep responses clear and concise.\n\n"
            for msg in st.session_state.messages:
                role_prefix = "Human: " if msg["role"] == "user" else "Assistant: "
                full_prompt += f"{role_prefix}{msg['content']}\n"
            full_prompt += "Assistant: "
            
            # Send the user's message to Ollama and get a streaming response
            response = llm.invoke(full_prompt)
            
            # Add assistant response to chat history - it was already displayed by the handler
            st.session_state.messages.append({"role": "assistant", "content": stream_handler.text})
        
        except Exception as e:
            # Handle errors gracefully
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add a button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()  # Rerun the app to update the UI
