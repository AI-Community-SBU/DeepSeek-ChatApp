import streamlit as st  # Import Streamlit library for creating web interfaces
from langchain_community.llms.ollama import Ollama  # Import Ollama integration from LangChain

# App title - this will appear at the top of the page
st.title("🤖 Model Comparison App")

# Add a brief description
st.markdown("Compare responses from different models side-by-side")

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

# Set up the sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    
    # Model A Selection
    st.subheader("Model A")
    model_a = st.selectbox(
        "Select first model",
        options=available_models,
        index=0 if available_models and len(available_models) > 0 else 0,
        key="model_a"
    )
    
    # Model B Selection
    st.subheader("Model B")
    # Default to second model in the list if available, otherwise use the first one
    default_model_b_index = 1 if len(available_models) > 1 else 0
    model_b = st.selectbox(
        "Select second model",
        options=available_models,
        index=default_model_b_index,
        key="model_b"
    )

# Initialize chat history in session state if it doesn't exist
if 'history_a' not in st.session_state:
    st.session_state.history_a = []
if 'history_b' not in st.session_state:
    st.session_state.history_b = []

# Create two columns for displaying chat history
col1, col2 = st.columns(2)

with col1:
    st.header(f"Model A: {model_a}")
    # Display chat history for Model A
    for message in st.session_state.history_a:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.header(f"Model B: {model_b}")
    # Display chat history for Model B
    for message in st.session_state.history_b:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input for user to type messages - using a form for better control
st.divider()
prompt = st.chat_input("Ask both models a question...")

# If the user sends a message
if prompt:
    # Add user message to both chat histories and display it
    st.session_state.history_a.append({"role": "user", "content": prompt})
    st.session_state.history_b.append({"role": "user", "content": prompt})
    
    # Update the display for both columns with the new user message
    with col1:
        with st.chat_message("user"):
            st.markdown(prompt)
    with col2:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Process responses from both models side-by-side
    with col1:
        with st.chat_message("assistant"):
            with st.spinner(f"Model A ({model_a}) is thinking..."):
                try:
                    # Initialize Model A
                    llm_a = Ollama(model=model_a)
                    
                    # Get response from Model A
                    response_a = llm_a.invoke(prompt)
                    
                    # Display the response from Model A
                    st.markdown(response_a)
                    
                    # Add assistant response to Model A chat history
                    st.session_state.history_a.append({"role": "assistant", "content": response_a})
                
                except Exception as e:
                    # Handle errors for Model A
                    error_msg = f"Error with {model_a}: {str(e)}"
                    st.error(error_msg)
                    st.session_state.history_a.append({"role": "assistant", "content": error_msg})
    
    with col2:
        with st.chat_message("assistant"):
            with st.spinner(f"Model B ({model_b}) is thinking..."):
                try:
                    # Initialize Model B
                    llm_b = Ollama(model=model_b)
                    
                    # Get response from Model B
                    response_b = llm_b.invoke(prompt)
                    
                    # Display the response from Model B
                    st.markdown(response_b)
                    
                    # Add assistant response to Model B chat history
                    st.session_state.history_b.append({"role": "assistant", "content": response_b})
                
                except Exception as e:
                    # Handle errors for Model B
                    error_msg = f"Error with {model_b}: {str(e)}"
                    st.error(error_msg)
                    st.session_state.history_b.append({"role": "assistant", "content": error_msg})

# Add a button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.history_a = []
    st.session_state.history_b = []
    st.rerun()  # Rerun the app to update the UI

# Add helpful metrics comparison section
with st.expander("Compare Response Metrics", expanded=False):
    if len(st.session_state.history_a) > 0 and len(st.session_state.history_b) > 0:
        # Find the latest responses
        latest_a = next((msg["content"] for msg in reversed(st.session_state.history_a) 
                         if msg["role"] == "assistant"), None)
        latest_b = next((msg["content"] for msg in reversed(st.session_state.history_b) 
                         if msg["role"] == "assistant"), None)
        
        if latest_a and latest_b:
            # Simple metrics comparison
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.subheader(f"{model_a} Metrics")
                st.write(f"Response Length: {len(latest_a)} characters")
                st.write(f"Word Count: {len(latest_a.split())}")
            
            with metrics_col2:
                st.subheader(f"{model_b} Metrics")
                st.write(f"Response Length: {len(latest_b)} characters")
                st.write(f"Word Count: {len(latest_b.split())}")
