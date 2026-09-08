import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(
    page_title="Built HR Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Built HR Policy Assistant")
st.caption("Ask questions about company HR rules, leave policies, and procedures.")

# Sidebar - API Credentials
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter OpenRouter API Key:", type="password")
api_base = st.sidebar.text_input("API Base URL:", value="https://openrouter.ai/api/v1")

# Ensure API key is configured
if not api_key:
    st.info("👈 Please enter your OpenRouter / OpenAI-compatible API Key in the sidebar to proceed.")
    st.stop()

# Cache Vectorstore Initialization
@st.cache_resource(show_spinner="Processing HR Policy Document...")
def load_and_vectorize_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Chunk PDF content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(documents)
    
    # Generate Local Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Store in memory ChromaDB
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# Path to HR Policy file
DATA_PDF = "data/hr_policy.pdf"

if os.path.exists(DATA_PDF):
    vectorstore = load_and_vectorize_pdf(DATA_PDF)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    st.error(f"Missing PDF document at `{DATA_PDF}`. Please check directory structure.")
    st.stop()

# Instantiate Chat Model (openai/gpt-oss-20b)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_key=api_key,
    openai_api_base=api_base,
    temperature=0.2
)

# System Prompt Template
system_prompt = (
    "You are an expert HR Policy Assistant. Use the provided context below to answer "
    "the user's questions clearly, accurately, and professionally. If the answer cannot "
    "be found in the context, state clearly that the document does not mention it.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# RAG Chain Creation
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything regarding company policies, leave structure, or workplace conduct."}
    ]

# Render existing messages
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# Handle user input
if user_query := st.chat_input("e.g., What is the casual leave policy?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR Policy..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.write(answer)
                
                # Show source excerpts in expander
                with st.expander("View Reference Policy Clauses"):
                    for doc in response["context"]:
                        st.markdown(f"**Page {doc.metadata.get('page', 'N/A') + 1}:**")
                        st.write(doc.page_content)
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
