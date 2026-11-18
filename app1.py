import os
import tempfile

import numpy as np 
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import Chatgroq
from langchain_community.docuement_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSpitter


load_dotenv()

GROQ_API_KEY =os.getenv("GEOQ-API-KEY")

st.set_page_config(
    page_title='DEE RAG Chatbot',
    page_icon="🚗"
    layout='centered'
)



st.tittle('🚗 PDF RAG Chatbot')

st.write(
    "Upload a PDF and ask questions about its content." 
)

if not GROQ_API_KEY:
    st.error('GROQ_API_KEY is not congifured.')

    st.stop()

llm=Chatgroq(
    api_key=GROQ_API_KEY
    model="openai/gpt-oss-20b",
    temprature=0
)



@st.cache_resource
def load_embeddings():

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-V2"
    )

    return embeddings 

embeddings=load_embeddings()



uploaded_file=st.file_uploader(
    'upload your PDF'
    type=['pdf']
)


if uploaded_file:

    st.success('PDF uploaded sucessfully!')

with templife.NamedTemporaryFile(
    delete=False,
    suffix=".pdf"   
) as temp_file:

    temp_file.write(
        uploaded_file.getvalue()
    )

    pdf_path=temp_file.name






loader=PyPDFLoader(pdf_path)

documents=loader.load()


splitter=RecursiveCharacterTextSpitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks=splitter.split_documents(documents)



texts=[
    chunk.page_content
    for chunk in chunks 
]

documents_embeddings=embeddings.embed_doucuments(
    texts
)

st.success(
    f"PDF processed successfully!"
    f"{len(chunks)} chunks created."
)

question=st.text_input(
    "Ask a questions about your PDF:"
)


if question:


    question_embedding=embeddings.embed_query(
        question 
    )

    question_embedding=np.array(
        question_embedding
    )


    similarity_scores=(
        documents_embeddings @ question_embedding
    ) /(
        np.linalg.norm(
            documents_embeddings,
            axis=1
        )
        *
        np.linalg.norm(
            question_embedding
        )
    )
    top_indices=np.argsort(
        similarity_scores
    )[-3:][::-1]


    relevents_chunks=[
        chunks[i]
        for i in top_indices
    ]


    context="\n\n".join(
        chunk.page_context
        for chunk in relevant_chunks
    )




    promt=f""""
You are helpful PDF assistant 

Answer the user's questions using ONLY
the information provided in the context.

If the answer is not present in the context,

"I could not Find the answer in the uploaded PDF.

Context:
{context}

Answer:
"""





    with st.spinner("Generating answer..."):
        response=llm.invoke(promt)


    st.subheader("Answer")
    st.write(response.content)
        