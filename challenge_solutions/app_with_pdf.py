import streamlit as st
import os
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Create uploads directory if it doesn't exist
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# App title
st.title("🤖 Chat with DeepSeek + PDF")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'document_chain' not in st.session_state:
    st.session_state.document_chain = None

# Create a dropdown to select from available Ollama models
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        available_models = [model["name"] for model in response.json()["models"]]
    else:
        available_models = ["deepseek-coder", "deepseek-r1:1.5b"]
except Exception:
    st.warning("⚠️ Couldn't connect to Ollama API. Is Ollama running?")
    available_models = ["deepseek-coder", "deepseek-r1:1.5b"]

# Sidebar for settings and PDF upload
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Select a model",
        options=available_models,
        index=0 if available_models else 0,
    )
    
    st.header("📁 Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF to chat with", type="pdf")
    
    if uploaded_pdf:
        # Save the uploaded PDF
        pdf_path = os.path.join(UPLOADS_DIR, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        # Process the PDF
        with st.spinner("Processing PDF..."):
            try:
                # Load the PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings()
                vector_store = FAISS.from_documents(splits, embeddings)
                
                # Create retrieval chain
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                st.session_state.document_chain = ConversationalRetrievalChain.from_llm(
                    llm=Ollama(model=selected_model),
                    retriever=retriever,
                    return_source_documents=True
                )
                st.session_state.pdf_processed = True
                st.success(f"PDF processed: {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user to type messages
prompt = st.chat_input("Ask about the PDF...")

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
                if st.session_state.pdf_processed and st.session_state.document_chain:
                    # Format chat history for the retrieval chain
                    chat_history = [(msg["content"], st.session_state.messages[i+1]["content"]) 
                                  for i, msg in enumerate(st.session_state.messages[:-1]) 
                                  if msg["role"] == "user" and i+1 < len(st.session_state.messages) and
                                  st.session_state.messages[i+1]["role"] == "assistant"]
                    
                    # Use the retrieval chain to answer based on the PDF
                    response = st.session_state.document_chain.invoke({
                        "question": prompt,
                        "chat_history": chat_history
                    })
                    answer = response["answer"]
                else:
                    # If no PDF is loaded, use regular chat
                    llm = Ollama(model=selected_model)
                    answer = llm.invoke(f"User question: {prompt}\nAnswer: ")
                
                # Display the response
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                # Handle errors gracefully
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add a button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
