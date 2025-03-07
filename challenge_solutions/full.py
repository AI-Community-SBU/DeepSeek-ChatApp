import streamlit as st
import os
import json
import time
from datetime import datetime
import re

# Updated imports using latest LangChain modules
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain

# Directory setup
CHAT_HISTORY_DIR = "chat_histories"
UPLOADS_DIR = "uploads"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Define color palette with improved contrast
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #F8F9FA;
        color: #212529;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #2C2F33 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        font-size: 16px !important;
    }

    /* Button Styling - Improved visibility */
    .stButton>button {
        background-color: #0275d8 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
    }
    
    /* "Clear Context" Button - Different color */
    button[kind="secondary"] {
        background-color: #6c757d !important;
    }
    
    /* "Save Chat" Button - Different color */
    button[data-testid="baseButton-secondary"] {
        background-color: #28a745 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: bold;
    }

    /* Chat container */
    .chat-container {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }

    .user-message {
        background-color: #007BFF;
        color: white !important;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }

    .assistant-message {
        background-color: #E9ECEF;
        color: #212529 !important;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: left;
        max-width: 80%;
        margin-right: auto;
    }
    
    /* File Uploader */
    .stFileUploader>div>div>div>button {
        background-color: #FFC107;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
    }

    /* Fix Navigation Bar (Top Bar) */
    header {
        background-color: #1E1E1E !important;
    }
    header * {
        color: #FFFFFF !important;
    }
    
    /* Scrollable chat list */
    .scrollable-chat-list {
        height: 200px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'document_chain' not in st.session_state:
    st.session_state.document_chain = None
if 'available_chats' not in st.session_state:
    st.session_state.available_chats = []

# Function to create a new chat (extracted for reuse)
def new_chat():
    st.session_state.chat_history = []
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.pdf_processed = False
    st.session_state.document_chain = None

# Generate a clean filename from text
def clean_filename(text):
    # Remove special characters and limit length
    clean = re.sub(r'[^\w\s-]', '', text)[:30]
    # Replace spaces with underscores
    return re.sub(r'[-\s]+', '_', clean).strip('-_')

# Get available Ollama models
@st.cache_data(ttl=60)
def get_available_models():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            return ["deepseek-r1:1.5b"]  # Default fallback
    except Exception as e:
        st.warning(f"Could not connect to Ollama API: {e}")
        return ["deepseek-r1:1.5b"]  # Default fallback

# Load chat history from file
def load_chat_history(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
        return []

# Save chat history to file
def save_chat_history(conversation_id, title, messages):
    try:
        if not title:
            title = f"Chat_{conversation_id}"
        
        filename = f"{clean_filename(title)}_{conversation_id}.json"
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        
        with open(filepath, 'w') as file:
            json.dump(messages, file)
        
        return filepath
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")
        return None

# Get list of available chat histories
def get_available_chats():
    try:
        files = os.listdir(CHAT_HISTORY_DIR)
        chat_files = [f for f in files if f.endswith('.json')]
        # Extract title and ID from filenames
        chat_info = []
        for file in chat_files:
            name = file.rsplit('.', 1)[0]
            parts = name.rsplit('_', 1)
            if len(parts) > 1:
                title = parts[0].replace('_', ' ')
                chat_id = parts[1]
                chat_info.append({"title": title, "id": chat_id, "filename": file})
            else:
                chat_info.append({"title": name, "id": name, "filename": file})
        return chat_info
    except Exception as e:
        st.error(f"Failed to get available chats: {e}")
        return []

# Process uploaded PDF
def process_pdf(file_path):
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(splits, embeddings)
        
        # Create retrieval chain retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        return retriever
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# App title
st.title("🤖 ChatGPT-like Interface with Ollama")

# Sidebar for chat management and settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select Ollama Model", 
        options=available_models,
        index=available_models.index("deepseek-r1:1.5b") if "deepseek-r1:1.5b" in available_models else 0
    )
    
    # PDF uploader
    st.header("📁 Upload PDF Document")
    uploaded_pdf = st.file_uploader("Upload a PDF to chat with", type="pdf")
    
    if uploaded_pdf:
        # Save the file
        pdf_filename = os.path.join(UPLOADS_DIR, uploaded_pdf.name)
        with open(pdf_filename, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        # Process the PDF
        with st.spinner("Processing PDF..."):
            retriever = process_pdf(pdf_filename)
            if retriever:
                # Set PDF context only for the current chat
                st.session_state.document_chain = ConversationalRetrievalChain.from_llm(
                    llm=Ollama(model=selected_model),
                    retriever=retriever,
                    return_source_documents=True
                )
                st.session_state.pdf_processed = True
                st.success(f"PDF processed: {uploaded_pdf.name}")
    
    # Chat management
    st.header("Chat Management")
    
    # New chat button: Clear all chat context and PDF context
    if st.button("➕ New Chat", key="new_chat_btn"):
        new_chat()
        st.rerun()
    
    # Load saved chats as a scrollable list that auto-loads when clicked
    st.session_state.available_chats = get_available_chats()
    if st.session_state.available_chats:
        st.subheader("Load Previous Chat")
        st.markdown('<div class="scrollable-chat-list">', unsafe_allow_html=True)
        chat_options = {f"{chat['title']} ({chat['id']})": chat for chat in st.session_state.available_chats}
        selected_chat_key = st.radio("Select a chat", options=list(chat_options.keys()))
        st.markdown('</div>', unsafe_allow_html=True)
        selected_chat = chat_options[selected_chat_key]
        # Auto-load chat if it's not the current conversation
        if selected_chat["id"] != st.session_state.conversation_id:
            file_path = os.path.join(CHAT_HISTORY_DIR, selected_chat["filename"])
            loaded_history = load_chat_history(file_path)
            if loaded_history:
                st.session_state.chat_history = loaded_history
                st.session_state.conversation_id = selected_chat["id"]
                # Clear PDF context when loading a chat
                st.session_state.pdf_processed = False
                st.session_state.document_chain = None
                st.rerun()

# Main chat interface
chat_col1, chat_col2 = st.columns([5, 1])
with chat_col1:
    st.header("Chat")
with chat_col2:
    # Clear context button with improved styling
    if st.button("🗑️ Clear", key="clear_context", type="secondary"):
        new_chat()
        st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""<div class="user-message">{message["content"]}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="assistant-message">{message["content"]}</div>""", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    st.markdown(f"""<div class="user-message">{user_input}</div>""", unsafe_allow_html=True)
    
    # Process with Ollama
    with st.spinner("Thinking..."):
        try:
            if st.session_state.pdf_processed and st.session_state.document_chain:
                # Format chat history for RAG chain
                chat_history = [(msg["content"], st.session_state.chat_history[i+1]["content"]) 
                               for i, msg in enumerate(st.session_state.chat_history[:-1]) 
                               if msg["role"] == "user" and i+1 < len(st.session_state.chat_history) and
                               st.session_state.chat_history[i+1]["role"] == "assistant"]
                
                # Using RAG with PDF
                response = st.session_state.document_chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history
                })
                bot_response = response["answer"]
            else:
                # Regular chat without PDF
                ollama = Ollama(model=selected_model)
                # Construct prompt with chat history
                full_prompt = "You are a helpful AI assistant. Keep responses clear and concise.\n\n"
                for msg in st.session_state.chat_history:
                    prefix = "Human: " if msg["role"] == "user" else "AI: "
                    full_prompt += f"{prefix}{msg['content']}\n"
                full_prompt += "AI: "

                # Use invoke() instead of __call__
                bot_response = ollama.invoke(full_prompt)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            # Display bot response
            st.markdown(f"""<div class="assistant-message">{bot_response}</div>""", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})

# Chat title and save controls (chat is only saved when the button is clicked)
if st.session_state.chat_history:
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        chat_title = st.text_input("💾 Chat Title", 
                                  value=f"Chat from {datetime.now().strftime('%Y-%m-%d')}" if len(st.session_state.chat_history) <= 2 else "")
    with col2:
        if st.button("Save Chat", type="primary"):
            saved_path = save_chat_history(
                st.session_state.conversation_id,
                chat_title,
                st.session_state.chat_history
            )
            if saved_path:
                st.success("Chat saved successfully!")
                # Refresh the list of available chats
                st.session_state.available_chats = get_available_chats()
