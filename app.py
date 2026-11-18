# ==========================================
# PDF RAG CHATBOT
# Streamlit + LangChain + Groq
# ==========================================

# ------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------

import os
import tempfile

import numpy as np
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------
# 2. LOAD ENVIRONMENT VARIABLES
# ------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ------------------------------------------
# 3. STREAMLIT PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# ------------------------------------------
# 4. APP TITLE
# ------------------------------------------

st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload a PDF and ask questions about its content."
)


# ------------------------------------------
# 5. CHECK GROQ API KEY
# ------------------------------------------

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")

    st.stop()


# ------------------------------------------
# 6. LOAD GROQ MODEL
# ------------------------------------------

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0
)


# ------------------------------------------
# 7. LOAD EMBEDDING MODEL
# ------------------------------------------

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


embeddings = load_embeddings()


# ------------------------------------------
# 8. PDF UPLOAD
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ------------------------------------------
# 9. PROCESS PDF
# ------------------------------------------

if uploaded_file:

    st.success("PDF uploaded successfully!")

    # --------------------------------------
    # Save uploaded PDF temporarily
    # --------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        pdf_path = temp_file.name


    # --------------------------------------
    # LOAD PDF
    # --------------------------------------

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # --------------------------------------
    # SPLIT DOCUMENT INTO CHUNKS
    # --------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)


    # --------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    document_embeddings = embeddings.embed_documents(
        texts
    )

    document_embeddings = np.array(
        document_embeddings
    )


    st.success(
        f"PDF processed successfully! "
        f"{len(chunks)} chunks created."
    )


    # --------------------------------------
    # USER QUESTION
    # --------------------------------------

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    # --------------------------------------
    # RAG PIPELINE
    # --------------------------------------

    if question:

        # ----------------------------------
        # CREATE QUESTION EMBEDDING
        # ----------------------------------

        question_embedding = embeddings.embed_query(
            question
        )

        question_embedding = np.array(
            question_embedding
        )


        # ----------------------------------
        # CALCULATE COSINE SIMILARITY
        # ----------------------------------

        similarity_scores = (
            document_embeddings @ question_embedding
        ) / (
            np.linalg.norm(
                document_embeddings,
                axis=1
            )
            *
            np.linalg.norm(
                question_embedding
            )
        )


        # ----------------------------------
        # GET TOP 3 RELEVANT CHUNKS
        # ----------------------------------

        top_indices = np.argsort(
            similarity_scores
        )[-3:][::-1]


        relevant_chunks = [
            chunks[i]
            for i in top_indices
        ]


        # ----------------------------------
        # CREATE CONTEXT
        # ----------------------------------

        context = "\n\n".join(
            chunk.page_content
            for chunk in relevant_chunks
        )


        # ----------------------------------
        # CREATE PROMPT
        # ----------------------------------

        prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer is not present in the context,
say:

"I could not find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""


        # ----------------------------------
        # SEND TO GROQ
        # ----------------------------------

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)


        # ----------------------------------
        # DISPLAY ANSWER
        # ----------------------------------

        st.subheader("Answer")

        st.write(response.content) 