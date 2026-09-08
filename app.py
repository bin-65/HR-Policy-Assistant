import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="HR Policy Assistant", page_icon="👨‍💼")
st.title("👨‍💼 HR Policy Assistant")
st.caption("RAG-powered assistant for answering questions from an HR policy PDF.")

PDF_PATH = Path("data/hr_policy.pdf")

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()

@st.cache_resource
def build_vector_store():
    if not PDF_PATH.exists():
        st.error("data/hr_policy.pdf was not found.")
        st.stop()

    docs = PyPDFLoader(str(PDF_PATH)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

with st.spinner("Loading HR policy..."):
    vector_store = build_vector_store()

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

question = st.text_input(
    "Ask your HR policy question:",
    placeholder="Example: How many annual leaves are allowed?"
)

if st.button("Ask HR Assistant", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = ChatPromptTemplate.from_template("""
You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy context below.
Do not invent or guess company rules.

If the answer is not contained in the context, say:
"I could not find this information in the HR policy."

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""")

        with st.spinner("Finding the policy and generating answer..."):
            response = llm.invoke(
                prompt.format(context=context, question=question)
            )

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("View retrieved policy sources"):
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page")
                page_text = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
                st.markdown(f"**Source {i} — {page_text}**")
                st.write(doc.page_content)
