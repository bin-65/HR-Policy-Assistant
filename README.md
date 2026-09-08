# HR Policy Assistant — RAG

A beginner-friendly Streamlit RAG application using:
- Groq LLM
- LangChain
- FAISS vector search
- Hugging Face sentence-transformers embeddings
- PDF HR policy

## Project structure

```text
hr-policy-rag/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── hr_policy.pdf
```

## Important

Do NOT upload your API key to GitHub.

For local testing, create:

`.streamlit/secrets.toml`

with:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

For Streamlit Community Cloud, add the same secret in the app's Secrets settings.

## HR PDF

Replace `data/hr_policy.pdf` with your own HR policy PDF.
