"""
Ingest Checkmate data from Google Sheets.
- Only keep rows where the Name column has data.
- Stop when reaching a row that starts with "Total".
- Create embeddings and store them in ChromaDB for RAG.
"""

import os
import json
import hashlib
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# =========================
# CONFIG
# =========================
SHEET_KEY = os.getenv("SHEET_KEY")
WORKSHEET_INDEX = 0
CREDENTIALS_PATH = "config/credentials.json"

CHROMA_DIR = "data/chroma_db_checkmate"
COLLECTION_NAME = "checkmate_customers"
STATE_PATH = "data/checkmate_ingested_hashes.json"

# OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# =========================
# GOOGLE SHEET CONNECTION
# =========================
def connect_gsheet(credentials_path: str) -> gspread.Client:
    """Connect to Google Sheets API."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    return gspread.authorize(creds)


# =========================
# FETCH AND FILTER DATA
# =========================
def fetch_checkmate_data(
    client: gspread.Client, sheet_key: str, worksheet_index: int = 0
) -> pd.DataFrame:
    """
    Fetch data from Google Sheets with filtering:
    1. Keep rows where the Name column has data.
    2. Stop when reaching a row starting with "Total".
    """
    sh = client.open_by_key(sheet_key)
    ws = sh.get_worksheet(worksheet_index)
    values = ws.get_all_values()

    if not values or len(values) < 1:
        return pd.DataFrame()

    # Get headers (first row)
    headers = values[0]

    # Filter data rows
    filtered_rows = []
    for i, row in enumerate(values[1:], start=2):  # Data starts from row 2
        name_value = row[0].strip() if row and row[0] else ""

        # Stop if we encounter summary rows
        if name_value.lower().startswith("total"):
            print(f"Stopped at row {i}: found summary marker '{name_value}'")
            break

        if name_value:
            filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows, columns=headers)
    print(f"Fetched {len(df)} valid rows (before summary marker)")
    return df


# =========================
# CREATE DOCUMENTS FOR RAG
# =========================
def create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    """Convert DataFrame rows to LangChain documents for embedding."""
    documents = []

    total_run_col = (
        "Total Customer Spend"
        if "Total Customer Spend" in df.columns
        else "Tong Khach chay"
        if "Tong Khach chay" in df.columns
        else None
    )

    for idx, row in df.iterrows():
        name = row.get("Name", "")
        tech = row.get("Tech", "")
        account = row.get("Account", "")
        fee = row.get("Fee", "")
        balance = row.get("BALANCE", row.get("Balance", ""))
        total_run = row.get(total_run_col, "") if total_run_col else ""

        text_parts = [
            f"Customer: {name}",
            f"Assigned Tech: {tech}",
            f"Account: {account}",
            f"Fee: {fee}",
            f"Balance: {balance}",
        ]

        # Add daily spending data if present
        daily_data = []
        for col in df.columns:
            if "/" in col:  # Date columns like "1/2/2026"
                value = row.get(col, "")
                if value and value != "$0,00":
                    daily_data.append(f"{col}: {value}")

        if daily_data:
            text_parts.append(
                f"Daily spending: {', '.join(daily_data[:10])}"
            )  # Limit to first 10 days

        if total_run:
            text_parts.append(f"Total customer spend: {total_run}")

        text = "\n".join(text_parts)

        metadata = {
            "row_index": idx,
            "name": name,
            "tech": tech,
            "account": account,
            "balance": balance,
            "source": "checkmate_report",
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


# =========================
# HASH AND INCREMENTAL INGEST
# =========================
def compute_doc_hash(doc: Document) -> str:
    """Compute a hash for change detection."""
    content = doc.page_content + str(doc.metadata)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_state(state_path: str) -> Dict:
    """Load previously ingested document hashes."""
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state_path: str, state: Dict):
    """Save current document hashes."""
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# =========================
# CHROMADB OPERATIONS
# =========================
def ingest_documents(
    documents: List[Document], chroma_dir: str, collection_name: str, state_path: str
):
    """
    Ingest documents into ChromaDB with incremental updates.
    Only add/update changed documents.
    """
    old_state = load_state(state_path)
    new_state = {}

    embeddings = OpenAIEmbeddings()

    docs_to_add = []
    ids_to_add = []

    for doc in documents:
        doc_hash = compute_doc_hash(doc)
        row_idx = doc.metadata.get("row_index", -1)
        doc_id = f"doc_{row_idx}"

        new_state[str(row_idx)] = doc_hash

        if str(row_idx) not in old_state or old_state[str(row_idx)] != doc_hash:
            docs_to_add.append(doc)
            ids_to_add.append(doc_id)

    # Load or create ChromaDB
    if os.path.exists(chroma_dir):
        print(f"Loading existing ChromaDB from {chroma_dir}")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )

        if docs_to_add:
            print(f"Adding/updating {len(docs_to_add)} documents in ChromaDB")
            vectorstore.add_documents(documents=docs_to_add, ids=ids_to_add)
    else:
        if not docs_to_add:
            print("No documents to ingest")
            return None

        print(f"Creating new ChromaDB with {len(docs_to_add)} documents")
        vectorstore = Chroma.from_documents(
            documents=docs_to_add,
            embedding=embeddings,
            ids=ids_to_add,
            collection_name=collection_name,
            persist_directory=chroma_dir,
        )

    print(f"ChromaDB saved to {chroma_dir}")

    save_state(state_path, new_state)
    print(f"State saved to {state_path}")

    return vectorstore


# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("CHECKMATE DATA INGESTION")
    print("=" * 50)

    # 1. Connect to Google Sheets
    print("\n1. Connecting to Google Sheets...")
    client = connect_gsheet(CREDENTIALS_PATH)

    # 2. Fetch and filter data
    print("\n2. Fetching data from Google Sheets...")
    df = fetch_checkmate_data(client, SHEET_KEY, WORKSHEET_INDEX)

    if df.empty:
        print("No data found")
        return

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few customers:")
    balance_col = "BALANCE" if "BALANCE" in df.columns else "Balance"
    print(df[["Name", "Tech", "Account", balance_col]].head(10))

    # 3. Create documents
    print("\n3. Creating documents for embedding...")
    documents = create_documents_from_dataframe(df)
    print(f"Created {len(documents)} documents")

    # 4. Ingest into ChromaDB
    print("\n4. Ingesting documents into ChromaDB...")
    vectorstore = ingest_documents(documents, CHROMA_DIR, COLLECTION_NAME, STATE_PATH)

    # 5. Test retrieval
    print("\n5. Testing retrieval...")
    test_query = "What is the balance of customer C550?"
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"\nTest query: '{test_query}'")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:200])

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
