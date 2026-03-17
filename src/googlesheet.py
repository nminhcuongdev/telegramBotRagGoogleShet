"""
Google Sheet -> Ingest (handles blank/duplicate headers) -> Incremental Embedding -> FAISS -> LCEL RAG

Install:
  pip install gspread oauth2client pandas
  pip install langchain langchain-community langchain-openai faiss-cpu tiktoken

Env:
  export OPENAI_API_KEY="..."

Run:
  python sheet_rag.py
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple
from dotenv import load_dotenv

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =========================
# CONFIG
# =========================
SHEET_KEY = os.getenv("SHEET_KEY")
WORKSHEET_INDEX = 0  # 0 = first worksheet tab. Change index for another tab.
CREDENTIALS_PATH = "config/credentials.json"

FAISS_DIR = "data/faiss_index_sheet"
STATE_PATH = "data/ingested_hashes.json"

# Ignore auto-generated blank headers (__col_x) while embedding.
IGNORE_AUTO_COLS = True
AUTO_COL_PREFIX = "__col_"

# k trong retrieval
TOP_K = 50


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# =========================
# GOOGLE SHEET
# =========================
def connect_gsheet(credentials_path: str) -> gspread.Client:
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    return gspread.authorize(creds)


def normalize_headers(headers: List[str]) -> List[str]:
    """
    - blank header => __col_{index}
    - duplicate header => add _{n}
    """
    seen = {}
    out = []
    for idx, h in enumerate(headers):
        h = (h or "").strip()
        if h == "":
            h = f"{AUTO_COL_PREFIX}{idx+1}"

        if h in seen:
            seen[h] += 1
            h2 = f"{h}_{seen[h]}"
            out.append(h2)
        else:
            seen[h] = 0
            out.append(h)
    return out


def fetch_sheet_rows(client: gspread.Client, sheet_key: str, worksheet_index: int = 0) -> Tuple[List[str], List[Dict]]:
    sh = client.open_by_key(sheet_key)
    ws = sh.get_worksheet(worksheet_index)
    values = ws.get_all_values()

    if not values or len(values) < 1:
        return [], []

    raw_headers = values[0]
    data_rows = values[1:]

    headers = normalize_headers(raw_headers)

    rows = []
    for row in data_rows:
        # Pad row when columns are missing.
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        # Skip fully empty rows.
        if not any((c or "").strip() for c in row):
            continue
        rows.append(dict(zip(headers, row)))

    return headers, rows


# =========================
# INCREMENTAL STATE
# =========================
def row_hash(row: Dict) -> str:
    s = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_state(path: str) -> set:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_state(path: str, hashes: set) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(hashes)), f, ensure_ascii=False, indent=2)


# =========================
# DOC BUILDING
# =========================
def row_to_text(row: Dict) -> str:
    parts = []
    for k, v in row.items():
        if IGNORE_AUTO_COLS and k.startswith(AUTO_COL_PREFIX):
            continue
        v = "" if v is None else str(v)
        if v.strip() == "":
            continue
        parts.append(f"{k}: {v.strip()}")
    return "\n".join(parts).strip()


def build_documents(rows: List[Dict], sheet_key: str) -> List[Document]:
    docs: List[Document] = []
    # row_index: +2 because header is row 1.
    for i, r in enumerate(rows):
        text = row_to_text(r)
    
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "google_sheet",
                    "sheet_key": sheet_key,
                    "row_index": i + 2,
                },
            )
        )
    return docs


# =========================
# VECTORSTORE
# =========================
def load_or_create_vectorstore(embeddings: OpenAIEmbeddings, faiss_dir: str) -> FAISS:
    if os.path.exists(faiss_dir):
        return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    # Create an empty index by seeding with a small dummy document.
    dummy = [Document(page_content="init", metadata={"source": "init"})]
    vs = FAISS.from_documents(dummy, embeddings)
    vs.save_local(faiss_dir)
    return vs


def reset_vectorstore_with_docs(embeddings: OpenAIEmbeddings, docs: List[Document], faiss_dir: str) -> FAISS:
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(faiss_dir)
    return vs


# =========================
# LCEL RAG CHAIN (Modern Replacement)
# =========================
def build_rag_chain(vectorstore: FAISS):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Format docs with row_index for citation
    def format_docs(docs):
        return "\n\n".join(
            f"Row {d.metadata.get('row_index', 'N/A')}:\n{d.page_content}"
            for d in docs
        )

    # Custom prompt for precise row-based answers
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for querying Google Sheets data. Answer questions accurately.
        If you do not know the answer, say "I don't know" instead of guessing.
        Always cite the spreadsheet row numbers you used.
        Column definitions:
        - Name: Customer name or code
        - Tech: Assigned technical owner per row
        - Account: Assigned accountant per row
        - Fee: Monthly service fee per row
        - Balance: Current customer balance per row
        - Total Customer Spend: Total customer spend in the month.

        Sheet Rows:
        {context}

        Question: {question}

        Answer (cite rows):"""
    )

    # LCEL chain: retrieve -> format -> prompt -> LLM -> parse
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def build_rag_chain_full(vectorstore: FAISS, client: gspread.Client):
    """RAG chain that can query the full sheet when needed."""
    llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    def format_docs(docs):
        formatted = []
        for d in docs:
            row_idx = d.metadata.get('row_index', 'N/A')
            formatted.append(f"Row {row_idx}:\n{d.page_content}")
        return "\n".join(formatted)

    def get_all_rows_context():
        """Get all sheet data if the question needs full-table calculations."""
        headers, rows = fetch_sheet_rows(client, SHEET_KEY, WORKSHEET_INDEX)
        all_docs = build_documents(rows, SHEET_KEY)
        return "\n\n".join(
            f"Row {d.metadata.get('row_index', 'N/A')}:\n{d.page_content}"
            for d in all_docs
        )

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for querying Google Sheets data. Answer questions accurately.
        If you do not know the answer, say "I don't know" instead of guessing.
        Always cite the spreadsheet row numbers you used.
        Column definitions:
        - Name: Customer name or code
        - Tech: Assigned technical owner per row
        - Account: Assigned accountant per row
        - Fee: Monthly service fee per row
        - Balance: Current customer balance per row
        - Total Customer Spend: Total customer spend in the month.

Sheet Rows:
{context}

Question: {question}

Answer:"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# =========================
# INCREMENTAL INGEST
# =========================
def ingest_incremental(rows: List[Dict], vectorstore: FAISS, embeddings: OpenAIEmbeddings) -> int:
    ingested = load_state(STATE_PATH)

    new_docs = []
    new_hashes = []

    for i, r in enumerate(rows):
        h = row_hash(r)
        if h in ingested:
            continue
        doc_text = row_to_text(r)
        if not doc_text:
            continue
        new_docs.append(
            Document(
                page_content=doc_text,
                metadata={
                    "source": "google_sheet",
                    "sheet_key": SHEET_KEY,
                    "row_index": i + 2,
                },
            )
        )
        new_hashes.append(h)

    if new_docs:
        # add_documents performs embedding and inserts records into FAISS.
        vectorstore.add_documents(new_docs)
        vectorstore.save_local(FAISS_DIR)

        ingested.update(new_hashes)
        save_state(STATE_PATH, ingested)

    return len(new_docs)


# =========================
# MAIN PIPELINE
# =========================
def main():
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(f"Missing {CREDENTIALS_PATH}")

    # 1) read sheet
    client = connect_gsheet(CREDENTIALS_PATH)
    headers, rows = fetch_sheet_rows(client, SHEET_KEY, WORKSHEET_INDEX)

    print(f"Headers ({len(headers)}): {headers}")
    print(f"Rows: {len(rows)}")
    if not rows:
        print("No data rows found.")
        return

    # Optional: show sample dataframe
    df = pd.DataFrame(rows)
    print("\nSample data:")
    print(df.head(len(df)))

    # 2) embeddings + vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # If no FAISS index exists yet, create it from all documents.
    if not os.path.exists(FAISS_DIR):
        docs = build_documents(rows, SHEET_KEY)
        if not docs:
            print("No valid documents to embed.")
            return
        vectorstore = reset_vectorstore_with_docs(embeddings, docs, FAISS_DIR)
        # Build state hash from all ingested rows.
        all_hashes = {row_hash(r) for r in rows}
        save_state(STATE_PATH, all_hashes)
        print(f"Created new FAISS index with {len(docs)} docs.")
    else:
        vectorstore = load_or_create_vectorstore(embeddings, FAISS_DIR)
        added = ingest_incremental(rows, vectorstore, embeddings)
        print(f"Incremental ingest: added {added} docs.")

    # 3) LCEL RAG chain
    rag_chain = build_rag_chain(vectorstore)

    print("\n=== LCEL RAG Ready ===")
    print("Type your question (or 'exit'):")
    while True:
        q = input("\nQ> ").strip()
        if q.lower() in ("exit", "quit"):
            break

        answer = rag_chain.invoke(q)
        print("\nANSWER:\n", answer)


if __name__ == "__main__":
    main()
