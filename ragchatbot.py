import pdfplumber
import streamlit as st
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure Gemini API
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Store in .streamlit/secrets.toml

genai.configure(api_key=GEMINI_API_KEY)

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file and start asking questions",
        type="pdf"
    )

# Extract contents from PDF
if file is not None:

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store vectors in FAISS
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # User question
    user_question = st.text_input(
        "Type your question here"
    )

    if user_question:

        # Retrieve relevant chunks
        docs = retriever.invoke(user_question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Gemini prompt
        prompt = f"""
You are a helpful assistant answering questions about a PDF document.

Instructions:
1. Answer only from the provided context.
2. Give detailed and accurate answers.
3. If information is not available in the document,
   say "I could not find this information in the document."
4. Use bullet points when appropriate.

Context:
{context}

Question:
{user_question}

Answer:
"""

        # Gemini model
        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = model.generate_content(prompt)

        st.subheader("Answer")
        st.write(response.text)