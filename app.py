import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from tavily import TavilyClient
import os
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# STREAMLIT SESSION STATE INITIALIZATION
# ------------------------------------------------------------
# These variables store application state during the session.
# They allow the app to remember the loaded research paper,
# vector database, and metadata across user interactions.
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# ============================================================
# ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------
# This section loads API keys from the .env file and
# configures external services used in the system:
#
# 1. Gemini API → for LLM reasoning and generation
# 2. Tavily API → for real-time web search fallback
# ============================================================

# ENVIRONMENT
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

TAVILY_API_KEY = os.getenv("TAVILY_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ============================================================
# DATA PROCESSING: DOCUMENT LOADING
# ------------------------------------------------------------
# This function loads research papers from either:
# 1. Uploaded files (PDF or TXT)
# 2. A research paper URL
#
# LangChain document loaders convert raw sources into
# structured "Document" objects which contain:
# - page_content
# - metadata
#
# These documents will later be processed for embeddings
# and retrieval in the RAG pipeline.
# ============================================================

# DOCUMENT LOADING
def load_document(uploaded_file, url_input):

    if uploaded_file:

        if uploaded_file.type == "application/pdf":

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")

        else:

            with open("temp.txt", "wb") as f:
                f.write(uploaded_file.read())

            loader = TextLoader("temp.txt")

        documents = loader.load()
        return documents


    elif url_input:

        loader = WebBaseLoader(url_input)
        docs = loader.load()
        return docs


# ============================================================
# DATA PROCESSING: TEXT SPLITTING
# ------------------------------------------------------------
# Large documents cannot be embedded effectively as a single
# block of text. Therefore we split the document into smaller
# overlapping chunks.
#
# RecursiveCharacterTextSplitter ensures that:
# - Each chunk is around 1500 characters
# - 100 characters overlap between chunks
#
# Overlap helps preserve context between chunks and improves
# retrieval accuracy in the RAG pipeline.
# ============================================================

# TEXT SPLITTING
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )

    return text_splitter.split_documents(documents)


# ============================================================
# DATA PROCESSING: FULL TEXT EXTRACTION
# ------------------------------------------------------------
# This function concatenates the full content of the document
# into a single text string.
#
# The combined text is later used for:
# - Metadata extraction
# - Research paper summarization
# ============================================================

# EXTRACT FULL TEXT
def extract_text(documents):

    full_text = ""

    for doc in documents:
        full_text += doc.page_content + "\n"

    return full_text


# ============================================================
# DATA PROCESSING: METADATA EXTRACTION USING LLM
# ------------------------------------------------------------
# This function uses Gemini to extract structured metadata
# from the research paper.
#
# Extracted fields include:
# - Title
# - Authors
# - Abstract
# - Publication year
# - References
#
# Only the first portion of the paper is sent to the LLM
# to reduce token usage while still capturing key metadata.
# ============================================================

# METADATA EXTRACTION
def extract_metadata_llm(full_text):

    model = genai.GenerativeModel("gemini-flash-lite-latest")

    # Only send the beginning of the document to reduce token usage
    text_sample = full_text[:6000]

    prompt = f"""
You are an AI system that extracts research paper metadata.

From the text below extract:

Title:
Authors:
Abstract:
Published Year:
References (only research papers cited in the document, list 5-10 titles)

Return the result strictly in this format:

Title: ...
Authors: ...
Abstract: ...
Published Year: ...
References:
- ...
- ...
- ...

TEXT:
{text_sample}
"""

    response = model.generate_content(prompt)

    result = response.text

    # Basic parsing
    metadata = {
        "title": "Unknown",
        "authors": "Unknown",
        "abstract": "Not found",
        "year": "Unknown",
        "references": []
    }

    lines = result.split("\n")

    for line in lines:

        if line.lower().startswith("title:"):
            metadata["title"] = line.replace("Title:", "").strip()

        elif line.lower().startswith("authors:"):
            metadata["authors"] = line.replace("Authors:", "").strip()

        elif line.lower().startswith("abstract:"):
            metadata["abstract"] = line.replace("Abstract:", "").strip()

        elif line.lower().startswith("published year:"):
            metadata["year"] = line.replace("Published Year:", "").strip()

        elif line.strip().startswith("-"):

            ref = line.replace("-", "").strip()

            if len(ref) > 10:
                metadata["references"].append(ref)

    return metadata


# ============================================================
# REAL-TIME WEB SEARCH (FALLBACK KNOWLEDGE SOURCE)
# ------------------------------------------------------------
# If the answer cannot be found inside the research paper,
# the system performs a real-time web search using Tavily.
#
# Tavily returns the most relevant search results along with
# summaries and URLs, which are then formatted into readable
# content for the user.
# ============================================================

# WEB SEARCH
def web_search(query):

    results = tavily.search(query=query, max_results=3)

    content = "\n\n".join(
        [f"{r['title']}\n{r['content']}\nSource: {r['url']}" for r in results["results"]]
    )

    return content


# ============================================================
# RAG CHAIN (RETRIEVAL AUGMENTED GENERATION)
# ------------------------------------------------------------
# This function implements the core RAG pipeline:
#
# Step 1 → Retrieve relevant document chunks from FAISS
# Step 2 → Build context using retrieved chunks
# Step 3 → Send context + question to the LLM
# Step 4 → Generate a grounded answer
#
# If the answer does not exist in the retrieved context,
# the system triggers the web search fallback.
# ============================================================

# RAG ANSWER
def answer_question(query, vector_store):

    docs = vector_store.similarity_search(query, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    model = genai.GenerativeModel("gemini-flash-lite-latest")

    prompt = f"""
You are a research paper assistant.

Use ONLY the provided context.

If the answer does not exist in the context, return exactly:
NOT_FOUND

Context:
{context}

Question:
{query}

Answer:
"""

    response = model.generate_content(prompt)

    answer = response.text.strip()

    if "NOT_FOUND" in answer:
        return web_search(query), "web"

    return answer, "document"


# ============================================================
# RESEARCH PAPER SUMMARIZATION
# ------------------------------------------------------------
# This function generates a concise summary of the research
# paper using the Gemini model.
#
# Instead of summarizing the entire paper (which could be
# expensive in tokens), the system summarizes the abstract
# or a limited section of the document.
# ============================================================

# SUMMARY
def summarize_text(text):

    model = genai.GenerativeModel("gemini-flash-lite-latest")

    prompt = f"""
Summarize the following research paper clearly:

{text[:4000]}
"""

    response = model.generate_content(prompt)

    return response.text


# ============================================================
# STREAMLIT USER INTERFACE (UI FLOW)
# ------------------------------------------------------------
# This section defines the user interface and interaction
# flow of the application.
#
# Users can:
# - Upload research papers
# - Ask questions about the paper
# - Enable real-time web search
# - Generate paper summaries
# ============================================================

# STREAMLIT UI
st.set_page_config(page_title="Research Paper Management & Analysis Intelligence System")

st.title("Research Paper Management & Analysis Intelligence System")


# SIDEBAR
web_toggle = st.sidebar.toggle("Enable Real-Time Web Search")

# Upload area
if not st.session_state.documents_loaded:

    uploaded_file = st.file_uploader(
        "Upload Research Paper (PDF/TXT)", type=["pdf", "txt"]
    )

    url_input = st.text_input("Or Paste Research Paper URL")

else:

    st.success("Research paper already loaded")

    st.markdown("### Current Paper")
    st.write(st.session_state.paper_metadata["title"])

    uploaded_file = None
    url_input = None


# Question input 
query = st.text_input("Ask a question about the research paper")


# ============================================================
# DOCUMENT PROCESSING PIPELINE
# ------------------------------------------------------------
# When a research paper is uploaded:
#
# 1. Load the document
# 2. Extract full text
# 3. Extract metadata
# 4. Split text into chunks
# 5. Generate embeddings
# 6. Store embeddings in FAISS vector database
#
# The resulting vector store enables semantic retrieval
# for the RAG system.
# ============================================================

# PROCESS DOCUMENT
if (uploaded_file or url_input) and not st.session_state.documents_loaded:

    documents = load_document(uploaded_file, url_input)

    if documents:

        raw_text = extract_text(documents)

        metadata = extract_metadata_llm(raw_text)

        st.session_state.paper_metadata = metadata

        docs = split_documents(documents)

        # ====================================================
        # EMBEDDINGS GENERATION
        # ----------------------------------------------------
        # Convert text chunks into numerical vector
        # representations using Gemini embedding model.
        #
        # These embeddings capture semantic meaning,
        # allowing similarity-based retrieval later.
        # ====================================================

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # ====================================================
        # VECTOR DATABASE (FAISS)
        # ----------------------------------------------------
        # The embeddings and corresponding document chunks
        # are stored inside a FAISS vector database.
        #
        # FAISS enables fast similarity search to retrieve
        # the most relevant chunks during question answering.
        # ====================================================

        vector_store = FAISS.from_documents(docs, embeddings)

        st.session_state.vector_store = vector_store
        st.session_state.documents_loaded = True

        st.sidebar.success("Research paper loaded successfully")


# ============================================================
# QUERY ANSWERING FLOW
# ------------------------------------------------------------
# When a user submits a question:
#
# If Web Search is enabled:
# → The system performs a real-time web search
#
# Otherwise:
# → The RAG pipeline retrieves relevant document chunks
# → The LLM generates a context-grounded answer
# ============================================================

# ANSWER QUERY
if query and len(query.strip()) > 3:

    if web_toggle:

        with st.spinner("Searching the web..."):
            answer = web_search(query)

        st.subheader("Answer")
        st.write(answer)
        st.caption("Source: Web")

    else:

        vector_store = st.session_state.vector_store

        if vector_store:

            with st.spinner("Answering from research paper..."):
                answer, source = answer_question(query, vector_store)

            st.subheader("Answer")
            st.write(answer)

            if source == "document":
                st.caption("Source: Research Paper")
            else:
                st.caption("Source: Web")

        else:
            st.warning("Please upload a research paper first.")


# ============================================================
# SUMMARY GENERATION FLOW
# ------------------------------------------------------------
# This button allows users to generate a quick summary
# of the research paper using the extracted abstract.
# ============================================================

# SUMMARY BUTTON
if st.button("Generate Paper Summary"):

    metadata = st.session_state.get("paper_metadata", None)

    if metadata:

        with st.spinner("Generating summary..."):

            summary = summarize_text(metadata["abstract"])

            st.subheader("Summary")
            st.write(summary)



# activate virtual environment in terminal

# cd "C:\Users\User1\Desktop\Research Paper Project" (optional)

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\Activate.ps1


# changes into github code.
# git add .
# git commit -m "Added PDF summarization feature"
# git push