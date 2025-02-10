import os
import streamlit as st
import pickle
import time
import nltk
from langchain_community.llms import HuggingFaceHub
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Download NLTK data
nltk.download('punkt')

# Set up Streamlit interface
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# URL inputs
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "Faiss_Store_Openai.pkl"

main_placeholder = st.empty()

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=huggingface_token
)

if process_url_clicked:
    try:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
        
        main_placeholder.text("✅ Processing Complete! You can now ask questions about the articles.")
        
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")

# Question answering interface
query = st.text_input("Ask a question about the articles:")
if query:
    if os.path.exists(file_path):
        try:
            # Load the saved FAISS index
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            
            # Create QA chain
            chain = load_qa_with_sources_chain(
                llm=llm,
                chain_type="stuff",
                verbose=True
            )
            
            # Perform similarity search and get answer
            docs = vectorstore.similarity_search(query)
            result = chain(
                {"input_documents": docs, "question": query},
                return_only_outputs=True
            )
            
            # Display results
            st.header("Answer")
            st.write(result["output_text"])
            
            # Process and display sources if available
            if "Sources:" in result["output_text"]:
                sources_text = result["output_text"].split("Sources:")[1].strip()
                if sources_text:
                    st.subheader("Sources:")
                    for source in sources_text.split("\n"):
                        if source.strip():
                            st.write(source.strip())
                            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
    else:
        st.warning("Please process some articles first by adding URLs and clicking 'Process URLs'")

# Add a footer with instructions
st.sidebar.markdown("""
### How to use:
1. Paste URLs of news articles in the sidebar
2. Click 'Process URLs' to analyze them
3. Ask questions about the articles in the main window
""")

