import os
import re
import tempfile

import chromadb
import streamlit as st
from pypdf import PdfReader
from google import genai


# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Keep model names in one place.
# If Google changes a model name later, you only need to
# change it here.

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-2.5-flash"

COLLECTION_NAME = "hr_policies"
CHROMA_PATH = "./chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():
    """
    Create and cache the Gemini client.
    """

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_collection():
    """
    Create or retrieve the ChromaDB collection.
    """

    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        return collection

    except Exception as e:
        raise RuntimeError(
            f"Could not initialize ChromaDB: {e}"
        )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from every page of an uploaded PDF.

    Returns:
        List of dictionaries containing page number and text.
    """

    if uploaded_file is None:
        raise ValueError("No PDF file was provided.")

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        raise ValueError(
            f"{uploaded_file.name} is empty."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        reader = PdfReader(temp_path)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

                if text and text.strip():

                    pages.append(
                        {
                            "page": page_number,
                            "text": text
                        }
                    )

            except Exception as e:

                # Skip a problematic page rather than
                # crashing the entire application.
                print(
                    f"Could not read page "
                    f"{page_number} of "
                    f"{uploaded_file.name}: {e}"
                )

        if not pages:

            raise ValueError(
                f"No readable text was found in "
                f"{uploaded_file.name}. "
                "Make sure the PDF contains selectable text."
            )

        return pages

    except Exception as e:

        if isinstance(e, ValueError):
            raise

        raise ValueError(
            f"Could not read PDF "
            f"{uploaded_file.name}: {e}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean unnecessary whitespace from extracted PDF text.
    """

    if not text:
        return ""

    # Replace multiple spaces/newlines with one space.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages,
    filename,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split PDF text into overlapping chunks.

    Each chunk keeps:
    - text
    - source filename
    - page number
    """

    if overlap >= chunk_size:

        raise ValueError(
            "Chunk overlap must be smaller "
            "than chunk size."
        )

    chunks = []

    for page in pages:

        text = clean_text(
            page.get("text", "")
        )

        if not text:
            continue

        page_number = page.get(
            "page",
            "Unknown"
        )

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "source": filename,
                        "page": page_number
                    }
                )

            # Move forward while maintaining overlap.
            start += chunk_size - overlap

    return chunks


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def generate_embeddings(texts):
    """
    Generate one embedding for each text.

    IMPORTANT:
    We intentionally process one text at a time.

    This guarantees:

        8 texts -> 8 embeddings

    which prevents the ChromaDB error:

        Unequal lengths for fields:
        ids, metadatas, embeddings, documents
    """

    if not texts:

        raise ValueError(
            "No text was provided for embedding."
        )

    client = get_gemini_client()

    embeddings = []

    for index, text in enumerate(texts):

        if not text or not text.strip():
            raise ValueError(
                f"Empty text found at chunk {index}."
            )

        try:

            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )

            if not result.embeddings:

                raise ValueError(
                    "Gemini returned no embedding."
                )

            # One text -> first embedding.
            embedding = result.embeddings[0].values

            if not embedding:

                raise ValueError(
                    "Gemini returned an empty embedding."
                )

            embeddings.append(embedding)

        except Exception as e:

            raise RuntimeError(
                f"Failed to generate embedding "
                f"for chunk {index + 1}: {e}"
            )

    # Safety check before sending anything to Chroma.
    if len(embeddings) != len(texts):

        raise ValueError(
            "Embedding count does not match text count. "
            f"Texts: {len(texts)}, "
            f"Embeddings: {len(embeddings)}"
        )

    return embeddings


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(uploaded_files):
    """
    Process uploaded HR policy PDFs and store them in ChromaDB.

    Pipeline:

        PDF
        ↓
        Text extraction
        ↓
        Chunking
        ↓
        Embeddings
        ↓
        ChromaDB
    """

    if not uploaded_files:

        raise ValueError(
            "Please upload at least one PDF."
        )

    collection = get_collection()

    all_chunks = []

    # --------------------------------------------------------
    # Extract and chunk every PDF
    # --------------------------------------------------------

    for uploaded_file in uploaded_files:

        try:

            pages = extract_text_from_pdf(
                uploaded_file
            )

            chunks = create_chunks(
                pages,
                uploaded_file.name
            )

            if chunks:

                all_chunks.extend(chunks)

        except Exception as e:

            raise RuntimeError(
                f"Error processing "
                f"{uploaded_file.name}: {e}"
            )

    if not all_chunks:

        raise ValueError(
            "No usable text was found in the uploaded PDFs."
        )

    # --------------------------------------------------------
    # Prepare ChromaDB data
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    ids = [
        f"chunk_{i}"
        for i in range(len(all_chunks))
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": str(chunk["page"])
        }
        for chunk in all_chunks
    ]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        texts
    )

    # --------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------

    if not (
        len(ids)
        == len(texts)
        == len(metadatas)
        == len(embeddings)
    ):

        raise ValueError(
            "Data length mismatch before ChromaDB insert:\n"
            f"IDs: {len(ids)}\n"
            f"Documents: {len(texts)}\n"
            f"Metadata: {len(metadatas)}\n"
            f"Embeddings: {len(embeddings)}"
        )

    # --------------------------------------------------------
    # Remove previous documents
    # --------------------------------------------------------

    try:

        existing_count = collection.count()

        if existing_count > 0:

            collection.delete(
                where={}
            )

    except Exception as e:

        raise RuntimeError(
            f"Could not clear old ChromaDB data: {e}"
        )

    # --------------------------------------------------------
    # Add new documents
    # --------------------------------------------------------

    try:

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not save documents to ChromaDB: {e}"
        )

    return len(all_chunks)


# ============================================================
# EMBED USER QUESTION
# ============================================================

def embed_question(question):
    """
    Convert the user's question into an embedding.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    client = get_gemini_client()

    try:

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question.strip()
        )

        if not result.embeddings:

            raise ValueError(
                "Gemini returned no embedding "
                "for the question."
            )

        embedding = result.embeddings[0].values

        if not embedding:

            raise ValueError(
                "Gemini returned an empty "
                "question embedding."
            )

        return embedding

    except Exception as e:

        raise RuntimeError(
            f"Could not generate question embedding: {e}"
        )


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    number_of_results=TOP_K
):
    """
    Search ChromaDB for the most relevant HR policy chunks.
    """

    collection = get_collection()

    # --------------------------------------------------------
    # Check whether documents exist
    # --------------------------------------------------------

    try:

        collection_count = collection.count()

    except Exception as e:

        raise RuntimeError(
            f"Could not read ChromaDB: {e}"
        )

    if collection_count == 0:

        return [], []

    # Never request more documents than actually exist.
    number_of_results = min(
        number_of_results,
        collection_count
    )

    # --------------------------------------------------------
    # Generate question embedding
    # --------------------------------------------------------

    query_embedding = embed_question(
        question
    )

    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    try:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results
        )

    except Exception as e:

        raise RuntimeError(
            f"ChromaDB search failed: {e}"
        )

    documents = results.get(
        "documents",
        [[]]
    )

    metadatas = results.get(
        "metadatas",
        [[]]
    )

    # Chroma returns nested lists because multiple queries
    # can be submitted.
    documents = documents[0] if documents else []
    metadatas = metadatas[0] if metadatas else []

    return documents, metadatas


# ============================================================
# GENERATE ANSWER
# ============================================================

def ask_question(question):
    """
    Retrieve relevant policy information and ask Gemini
    to generate a grounded answer.
    """

    if not question or not question.strip():

        return (
            "Please enter a question about the HR policies.",
            []
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    documents, metadatas = retrieve_documents(
        question
    )

    if not documents:

        return (
            "I couldn't find relevant information "
            "in the uploaded HR policies.",
            []
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []
    sources = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        metadata = metadata or {}

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

    # Remove duplicate sources.
    sources = list(
        dict.fromkeys(sources)
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an HR Policy Assistant for ABC Technologies.

Your job is to answer questions using ONLY the HR policy
information provided in the context below.

IMPORTANT RULES:

1. Do not invent or guess company policies.
2. Do not use outside knowledge to answer the question.
3. If the answer cannot be found in the provided context,
   say:
   "I couldn't find this information in the uploaded HR policies."
4. Give a clear and concise answer.
5. Mention important conditions, limits, or exceptions
   when they are present in the policy.
6. If the policy gives a specific number of days, dates,
   limits, or requirements, state them accurately.
7. Do not provide legal advice.
8. Do not claim to be a human HR manager.
9. Do not make assumptions about an employee's situation.
10. At the end, provide a short "Source" line using the
    document and page information supplied in the context.

HR POLICY CONTEXT:

{context}

USER QUESTION:

{question.strip()}
"""

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    client = get_gemini_client()

    try:

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        answer = response.text

        if not answer:

            raise ValueError(
                "Gemini returned an empty response."
            )

        return answer.strip(), sources

    except Exception as e:

        raise RuntimeError(
            f"Could not generate HR assistant response: {e}"
        )

