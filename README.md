# DeepSeek Chat Application

![DeepSeek Chat](https://img.shields.io/badge/DeepSeek-Chat-blue)

A simple web interface built with Streamlit that lets you chat with DeepSeek AI models running locally through Ollama.

## 📋 Prerequisites

- [Python 3.7+](https://www.python.org/downloads/)
- [Ollama](https://ollama.ai/download) - For running AI models locally
- Required Python packages: `streamlit`, `langchain`, `langchain-community`, `requests`

## 🚀 Getting Started

### Step 1: Install Ollama

1. Visit [ollama.ai/download](https://ollama.ai/download)
2. Download the appropriate version for your operating system (Windows/Mac/Linux)
3. Install Ollama following the on-screen instructions

### Step 2: Install Required Python Packages

Open a terminal or command prompt and run:

```bash
pip install -r requirements.txt
```

### Step 3: Download DeepSeek Models (Optional)

You can pre-download models before running the app:

```bash
ollama pull deepseek-coder
ollama pull deepseek-r1:1.5b
```

## 🏃‍♂️ Running the Application

### Important: Start Ollama First!

⚠️ **CRITICAL STEP:** You must start Ollama in a terminal window before running the application!

1. Open a terminal or command prompt
2. Run the following command:

```bash
ollama start
```

3. Keep this terminal window open while using the application

### Starting the Chat Application

1. Open a new terminal or command prompt window
2. Navigate to the project directory:

```bash
cd "c:\Users\ruthv\OneDrive\Desktop\AI Community\DeepSeekCode"
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Your default web browser should open automatically with the application running (typically at http://localhost:8501)

## 🖥️ How to Use

1. Select a DeepSeek model from the dropdown menu in the sidebar
2. Type your question or prompt in the chat input box
3. View the AI's response in the chat window
4. Continue the conversation as needed
5. Use the "Clear Chat History" button in the sidebar to start a new conversation

## ⚠️ Troubleshooting

- **"Couldn't connect to Ollama API" error**: Make sure you ran `ollama start` in a separate terminal window and it's still running
- **Model not showing in dropdown**: You may need to download it first with `ollama pull model-name`
- **Slow responses**: Local AI models require significant resources - responses may take time based on your hardware
- **Application crashes**: Check that your system meets the minimum requirements to run DeepSeek models locally

## 💡 Available Models

The application automatically detects models installed through Ollama. Common DeepSeek models include:

- `deepseek-coder` - Optimized for programming tasks
- `deepseek-r1:1.5b` - Smaller, faster model with general capabilities
- `deepseek-llm` - General purpose model

You can install additional models using `ollama pull model-name`

## 🧩 Coding Challenges

Try these challenges to enhance your application and improve your skills with Streamlit, LangChain, and Ollama!

### 🔰 Beginner Challenges

#### 1. Add Temperature Control

**Description:** Add a slider to control the temperature parameter of the LLM responses.
**Explanation:** Temperature controls randomness - lower values make responses more deterministic, higher values more creative.

**Useful Functions:**

```python
# Streamlit slider
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# Ollama with temperature
llm = Ollama(model=selected_model, temperature=temperature)
```

#### 2. Message Character Counter

**Description:** Display a counter showing character count for user messages.
**Explanation:** This helps users understand how much input they're providing to the model.

**Useful Functions:**

```python
# Get character count
char_count = len(prompt)
st.sidebar.text(f"Characters: {char_count}")

# Or for advanced usage with changing colors
if char_count > 500:
    st.sidebar.text(f"Characters: {char_count} ⚠️")
```

### 🔄 Intermediate Challenges

#### 3. Chat Context Management

**Description:** Implement a function to limit the context sent to the model.
**Explanation:** LLMs have context limitations. Implement a sliding window of messages or token counting.

**Useful Functions:**

```python
# Get last N messages from history
def get_recent_context(messages, max_messages=10):
    return messages[-max_messages:] if len(messages) > max_messages else messages

# Example token counting with tiktoken
import tiktoken
def count_tokens(text, model="gpt-3.5-turbo"):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))
```

#### 4. Response Streaming

**Description:** Implement streaming for the model responses.
**Explanation:** Streaming shows responses character-by-character instead of waiting for the complete response.

**Useful Functions:**

```python
# Streaming with Ollama and Streamlit
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create placeholder
response_placeholder = st.empty()
collected_chunks = []

# Define custom handler
class StreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
      
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Use with Ollama
stream_handler = StreamHandler(response_placeholder)
llm = Ollama(model=selected_model, callbacks=[stream_handler])
response = llm.invoke(prompt)
```

### 🚀 Advanced Challenges

#### 5. Memory Management System

**Description:** Implement different memory options (conversation, summary, etc.).
**Explanation:** LangChain offers multiple memory types to manage conversation history differently.

**Useful Functions:**

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Simple conversation memory
memory = ConversationBufferMemory()

# Summary memory for longer conversations
memory = ConversationSummaryMemory(llm=llm)

# Use in Streamlit with session state
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
```

#### 6. Multi-Model Chat Comparison

**Description:** Allow users to compare responses from multiple models side-by-side.
**Explanation:** This helps users understand different model capabilities and behaviors.

**Useful Functions:**

```python
# Create columns
col1, col2 = st.columns(2)

# Get responses from multiple models
with col1:
    st.subheader("Model A Response")
    llm_a = Ollama(model="deepseek-coder")
    response_a = llm_a.invoke(prompt)
    st.markdown(response_a)
  
with col2:
    st.subheader("Model B Response")
    llm_b = Ollama(model="deepseek-llm")
    response_b = llm_b.invoke(prompt)
    st.markdown(response_b)
```

#### 7. Add RAG Capabilities

**Description:** Implement Retrieval Augmented Generation to enhance responses with external data.
**Explanation:** RAG improves responses by retrieving relevant documents before generating a response.

**Useful Functions:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# Load documents
loader = DirectoryLoader("./documents/", glob="**/*.txt")
documents = loader.load()

# Create embeddings and vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(chunks, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Use in Streamlit
response = qa_chain.invoke(prompt)
```

## 📚 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
- [DeepSeek AI](https://www.deepseek.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
