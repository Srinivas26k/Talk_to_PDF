import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import tempfile

# Add loading spinner
@st.cache_resource  # Cache the LLM and embeddings
def init_models():
    return OllamaLLM(
        model="llama3.2:latest",
        temperature=0.1,  # Lower temperature for more focused responses
        num_ctx=2048,    # Adjust context window
        num_thread=4     # Utilize multiple threads
    ), OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

# Initialize models with caching
ollama, embeddings = init_models()

# Optimize chunk settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Smaller chunks
    chunk_overlap=50,    # Less overlap
    length_function=len
)

# Set up Streamlit
st.title("PDF Chat with RAG")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process PDF when uploaded
if uploaded_file:
    with st.spinner('Processing PDF...'):
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

            # Load and process the PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Split text into chunks
            splits = text_splitter.split_documents(pages)

            # Create vector store
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.success("PDF processed successfully!")

# Chat interface
if st.session_state.vector_store:
    # Initialize the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ollama,
        retriever=st.session_state.vector_store.as_retriever(
            search_kwargs={"k": 3}  # Limit to top 3 relevant chunks
        ),
        return_source_documents=True,
        verbose=False  # Reduce logging
    )

    # Chat input
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        # Get the response
        try:
            response = qa_chain.invoke({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        # Update chat history
        st.session_state.chat_history.append((user_question, response["answer"]))

        # Clear old chat history if too long
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]

        # Display chat history
        for question, answer in st.session_state.chat_history:
            st.write(f"Q: {question}")
            st.write(f"A: {answer}")
            st.write("---")

    # Add a clear button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
    
    # Add file size limit
    if uploaded_file.size > 10_000_000:  # 10MB
        st.error("File too large. Please upload a smaller file.")
