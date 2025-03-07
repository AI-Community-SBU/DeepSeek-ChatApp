import streamlit as st  # Import Streamlit library for creating web interfaces
from langchain_community.llms.ollama import Ollama  # Import Ollama integration from LangChain

# App title - this will appear at the top of the page
st.title("🤖 Chat with DeepSeek")

# Add a brief description
st.markdown("This app implements context management to handle longer conversations")

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

# Function to get recent messages for context management
def get_recent_context(messages, max_messages=10):
    """Return only the most recent messages to manage context window"""
    return messages[-max_messages:] if len(messages) > max_messages else messages

# Try to import tiktoken for token counting if available
try:
    import tiktoken
    
    def count_tokens(text, model="gpt-3.5-turbo"):
        """Count the number of tokens in a text string"""
        try:
            encoder = tiktoken.encoding_for_model(model)
            return len(encoder.encode(text))
        except:
            # Fallback to approximate count if model not found
            return len(text) // 4  # Rough approximation
    
    has_tiktoken = True
except ImportError:
    # Fallback to character-based estimation if tiktoken not available
    def count_tokens(text, model=None):
        """Estimate tokens based on characters (rough approximation)"""
        return len(text) // 4
    
    has_tiktoken = False

# Create a dropdown in the sidebar to select the model
with st.sidebar:
    st.header("Model Settings")
    selected_model = st.selectbox(
        "Select a model",
        options=available_models,
        index=0 if available_models else 0,
        help="Choose which Ollama model to chat with"
    )
    
    # Context window settings
    st.header("Context Management")
    context_mode = st.radio(
        "Context Mode",
        ["Last N Messages", "Token Limit"],
        help="Choose how to manage conversation context"
    )
    
    if context_mode == "Last N Messages":
        max_messages = st.slider(
            "Maximum Messages in Context",
            min_value=2,
            max_value=20,
            value=10,
            step=1,
            help="Number of most recent messages to include in context"
        )
    else:
        max_tokens = st.slider(
            "Maximum Tokens in Context",
            min_value=500,
            max_value=8000,
            value=2000,
            step=100,
            help="Approximate token limit for context window"
        )
        if not has_tiktoken:
            st.info("Note: tiktoken not installed. Token counting is approximate.")

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
        with st.spinner("Thinking..."):
            try:
                # Get context-managed messages according to settings
                if context_mode == "Last N Messages":
                    context_messages = get_recent_context(st.session_state.messages, max_messages)
                else:
                    # Count tokens and include as many messages as fit within the limit
                    messages = []
                    token_count = 0
                    for msg in reversed(st.session_state.messages):
                        msg_tokens = count_tokens(msg["content"])
                        if token_count + msg_tokens <= max_tokens:
                            messages.insert(0, msg)
                            token_count += msg_tokens
                        else:
                            break
                    context_messages = messages
                
                # Construct prompt with limited context
                system_prompt = "You are a helpful AI assistant. Keep responses clear and concise.\n\n"
                for msg in context_messages:
                    prefix = "Human: " if msg["role"] == "user" else "AI: "
                    system_prompt += f"{prefix}{msg['content']}\n"
                system_prompt += "AI: "
                
                # Initialize Ollama with the selected model
                llm = Ollama(model=selected_model)
                
                # Send the context-limited prompt to Ollama
                response = llm.invoke(system_prompt)
                
                # Display the response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                # Handle errors gracefully
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add information about context management
with st.sidebar:
    st.subheader("Current Context Stats")
    total_messages = len(st.session_state.messages)
    st.write(f"Total Messages: {total_messages}")
    
    if context_mode == "Last N Messages":
        context_size = min(max_messages, total_messages)
        st.write(f"Messages in Context: {context_size}")
    else:
        if st.session_state.messages:
            # Calculate token count of all messages
            all_text = " ".join([msg["content"] for msg in st.session_state.messages])
            estimated_tokens = count_tokens(all_text)
            st.write(f"Estimated Total Tokens: {estimated_tokens}")
            st.write(f"Context Token Limit: {max_tokens}")
            if estimated_tokens > max_tokens:
                st.warning(f"Conversation exceeds token limit. Using most recent {max_tokens} tokens.")

# Add a button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()  # Rerun the app to update the UI
