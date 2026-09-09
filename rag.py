import os
import re
import tempfile

import chromadb
import streamlit as st
from pypdf import PdfReader
from google import genai


# -------------------------------------------------
# Gemini Client
# -------------------------------------------------

def get_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to Streamlit secrets."
        )

    return genai.Client(api_key=api_key)


# -------------------------------------------------
# ChromaDB
# -------------------------------------------------

@st.cache_resource
def get_collection():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="hr_policies"
    )

    return collection


# -------------------------------------------------
# Extract PDF text
# -------------------------------------------------

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:

        reader = PdfReader(temp_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )

        return pages

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------------------------------------
# Clean text
# -------------------------------------------------

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# -------------------------------------------------
# Split text into chunks
# -------------------------------------------------

def create_chunks(
    pages,
    filename,
    chunk_size=800,
    overlap=150
):

    chunks = []

    for page in pages:

        text = clean_text(page["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            if chunk.strip():

                chunks.append(
                    {
                        "text": chunk,
                        "source": filename,
                        "page": page["page"]
                    }
                )

            start += chunk_size - overlap

    return chunks


# -------------------------------------------------
# Generate embeddings
# -------------------------------------------------

def generate_embeddings(texts):

    client = get_gemini_client()

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=texts
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


# -------------------------------------------------
# Process uploaded documents
# -------------------------------------------------

def process_documents(uploaded_files):

    collection = get_collection()

    # Clear previous documents
    try:
        collection.delete(
            where={}
        )
    except Exception:
        pass

    all_chunks = []

    for uploaded_file in uploaded_files:

        pages = extract_text_from_pdf(
            uploaded_file
        )

        chunks = create_chunks(
            pages,
            uploaded_file.name
        )

        all_chunks.extend(chunks)

    if not all_chunks:

        raise ValueError(
            "No readable text was found in the uploaded PDFs."
        )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = generate_embeddings(
        texts
    )

    ids = [
        f"chunk_{i}"
        for i in range(len(all_chunks))
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(all_chunks)


# -------------------------------------------------
# Search relevant policy chunks
# -------------------------------------------------

def retrieve_documents(
    question,
    number_of_results=5
):

    collection = get_collection()

    client = get_gemini_client()

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    query_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    return documents, metadatas


# -------------------------------------------------
# Generate answer using Gemini
# -------------------------------------------------

def ask_question(question):

    documents, metadatas = retrieve_documents(
        question
    )

    if not documents:

        return (
            "I couldn't find relevant information "
            "in the uploaded HR policies.",
            []
        )

    context_parts = []

    sources = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        source = metadata.get(
            "source",
            "Unknown document"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{document}
"""
        )

        sources.append(
            f"{source} — Page {page}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question ONLY using
the HR policy context provided below.

Rules:

1. Do not invent company policies.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   clearly say that the information is not available
   in the uploaded policies.
4. Give a clear and simple answer.
5. If useful, use bullet points.
6. Mention important conditions or exceptions.
7. Do not pretend to be a lawyer or HR manager.
8. At the end, mention the relevant policy source.

HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}
"""

    client = get_gemini_client()

    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=prompt
    )

    answer = response.text

    return answer, list(dict.fromkeys(sources))
