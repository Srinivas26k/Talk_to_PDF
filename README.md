# PDF Chat with RAG (Retrieval-Augmented Generation)

A streamlined application that enables users to chat with their PDF documents using local Large Language Models through Ollama.

## 🚀 Features

- PDF document processing and analysis
- Real-time chat interface
- Local LLM usage (no data sent to external APIs)
- Memory-efficient document processing
- Session-based chat history
- User-friendly Streamlit interface

## 🛠️ Technical Stack

### Core Components
- **LLM**: Llama 3.2 (via Ollama)
- **Embeddings**: Nomic-embed-text
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **UI Framework**: Streamlit
- **PDF Processing**: PyPDF

### Key Libraries
- `langchain`: Framework for LLM applications
- `langchain-ollama`: Interface for Ollama models
- `faiss-cpu`: Vector similarity search
- `streamlit`: Web interface
- `pypdf`: PDF processing

## 📋 Prerequisites

1. **Python**: Version 3.8 or higher
2. **Ollama**: Must be installed and running
3. **Required Models**:
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text:latest
   ```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama service**
   ```bash
   ollama serve
   ```

4. **Run the application**
   ```bash
   streamlit run rag.py
   ```

## 💡 Usage Guide

1. **Upload PDF**
   - Click "Upload your PDF" button
   - Select any PDF file (recommended size < 10MB)
   - Wait for processing confirmation

2. **Chat Interface**
   - Type your question in the text input
   - Questions can be about any content in the uploaded PDF
   - System will retrieve relevant context and provide answers

3. **Chat History**
   - Previous Q&A pairs are displayed
   - History is maintained during the session
   - Use "Clear Chat" to reset conversation

## ⚙️ Technical Details

### Document Processing
- PDFs are split into chunks (500 tokens)
- Chunks overlap (50 tokens) to maintain context
- Embeddings are generated using nomic-embed-text
- Vectors stored in FAISS for efficient retrieval

### RAG Implementation
- Uses Conversational Retrieval Chain
- Retrieves top 3 most relevant chunks
- Combines retrieved context with user question
- Generates coherent responses using Llama 3.2

### Optimizations
- Model caching for better performance
- Efficient memory management
- Streamlined document processing
- Error handling and validation

## 🔒 Privacy & Security

- All processing happens locally
- No data sent to external servers
- PDFs are processed in memory
- Temporary files are automatically cleaned up

## ⚠️ Limitations

- PDF size limitations (recommended < 10MB)
- Processing time depends on hardware
- Requires local GPU/CPU resources
- Limited to text content in PDFs

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📝 License

[Your License Information](MIT)

## 🙏 Acknowledgments

- Langchain community
- Ollama project
- Streamlit team
- FAISS developers

## 📞 Support

[Your Support Information](https://in.linkedin.com/in/srinivas-nampalli)

---

*Note: This RAG implementation is designed for local deployment and requires appropriate hardware resources for optimal performance.* 
