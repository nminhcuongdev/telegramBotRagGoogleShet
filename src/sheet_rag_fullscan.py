#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Sheet -> Ingest -> Incremental Embedding -> FAISS -> LCEL RAG
+ FIX: Force FULL SCAN for statistical / aggregation questions (count / summary / list).

What changed vs original:
- Adds a simple router:
  - If question contains statistical keywords -> fetch ALL rows -> compute via pandas.
  - Else -> normal RAG retrieval over FAISS.
- Keeps row_index citations for both modes.

Run:
  python sheet_rag_fullscan.py
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Tuple, Optional
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
WORKSHEET_INDEX = 0  # 0 = first sheet tab. Change this for another tab.
CREDENTIALS_PATH = "config/credentials.json"

FAISS_DIR = "data/faiss_index_sheet"
STATE_PATH = "data/ingested_hashes.json"

# Ignore auto-generated blank headers (__col_x) while embedding.
IGNORE_AUTO_COLS = True
AUTO_COL_PREFIX = "__col_"

TOP_K = 10

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Keywords to trigger FULL SCAN (stats/aggregation)
STATS_KEYWORDS = [
    "statistics",
    "summary",
    "count",
    "how many",
    "total",
    "sum",
    "list",
    "total balance",
    "total customer spend",
]

# Core columns
COL_NAME = "Name"
COL_TECH = "Tech"
COL_ACCOUNT = "Account"
COL_FEE = "Fee"
COL_BALANCE = "Balance"
COL_TOTAL_RUN = "Total Customer Spend"

# Optional tech alias normalization
TECH_ALIAS = {
    "hoang": "Hoang",
    "hai": "Hai",
    "an": "An",
    "trung": "Trung",
}


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
            out.append(f"{h}_{seen[h]}")
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

    headers = normalize_headers(values[0])
    data_rows = values[1:]

    rows = []
    for row in data_rows:
        # Pad row if columns are missing.
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
    dummy = [Document(page_content="init", metadata={"source": "init"})]
    vs = FAISS.from_documents(dummy, embeddings)
    vs.save_local(faiss_dir)
    return vs


def reset_vectorstore_with_docs(embeddings: OpenAIEmbeddings, docs: List[Document], faiss_dir: str) -> FAISS:
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(faiss_dir)
    return vs


# =========================
# LCEL RAG CHAIN
# =========================
def build_rag_chain(vectorstore: FAISS):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    def format_docs(docs):
        return "\n\n".join(
            f"Row {d.metadata.get('row_index', 'N/A')}:\n{d.page_content}" for d in docs
        )

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for querying Google Sheets data. Answer questions accurately.
If you do not know the answer, say "I don't know" instead of guessing.
Always cite the spreadsheet row numbers you used.

Column definitions:
- Name: Customer name or code
- Tech: Assigned technical owner
- Account: Assigned accountant
- Fee: Monthly service fee
- Balance: Current customer balance
- Total Customer Spend: Total customer spending in the month

Sheet rows (retrieved):
{context}

Question: {question}

Answer (cite rows):"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# =========================
# FULL SCAN (deterministic stats)
# =========================
def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_key(s: str) -> str:
    return _norm_text(s).lower()


def _norm_tech(s: str) -> str:
    k = _norm_key(s)
    return TECH_ALIAS.get(k, _norm_text(s))


def parse_money_to_float(x: str) -> float:
    s = _norm_text(x)
    if s == "":
        return 0.0
    s = re.sub(r"[^0-9,\.\-]", "", s)
    if s in ("", "-", ".", ","):
        return 0.0

    if "." in s and "," in s:
        dec_pos = max(s.rfind("."), s.rfind(","))
        int_part = re.sub(r"[.,]", "", s[:dec_pos])
        frac_part = re.sub(r"[.,]", "", s[dec_pos + 1 :])
        s2 = int_part + ("." + frac_part if frac_part else "")
    elif "." in s:
        parts = s.split(".")
        s2 = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) <= 2 and len(parts) > 1 else "".join(parts)
    elif "," in s:
        parts = s.split(",")
        s2 = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) <= 2 and len(parts) > 1 else "".join(parts)
    else:
        s2 = s

    try:
        return float(s2)
    except Exception:
        return 0.0


def is_stats_question(q: str) -> bool:
    qk = _norm_key(q)
    return any(k in qk for k in STATS_KEYWORDS)


def extract_tech_name(question: str) -> Optional[str]:
    q = _norm_text(question)

    m = re.search(r"(?:tech|technical)\s*[:=\-]?\s*([A-Za-z\s\.]+)", q, flags=re.IGNORECASE)
    if m:
        return _norm_tech(" ".join(m.group(1).strip().split()[:3]))

    m = re.search(r"([A-Za-z\s\.]{2,30})\s+manages", q, flags=re.IGNORECASE)
    if m:
        return _norm_tech(" ".join(m.group(1).strip().split()[-3:]))

    return None


def full_scan_stats_answer(rows: List[Dict], question: str) -> str:
    """
    Exact stats over the full table. No vector retrieval, so no missing rows.
    Focus on Tech-based list/count/sum balance/fee/total run.
    """
    if not rows:
        return "I do not know (the sheet has no data)."

    df = pd.DataFrame(rows)
    df["_row_index"] = range(2, 2 + len(df))

    # Backward compatibility
    if "Tong Khach chay" in df.columns and COL_TOTAL_RUN not in df.columns:
        df.rename(columns={"Tong Khach chay": COL_TOTAL_RUN}, inplace=True)
    if "BALANCE" in df.columns and COL_BALANCE not in df.columns:
        df.rename(columns={"BALANCE": COL_BALANCE}, inplace=True)

    tech = extract_tech_name(question)
    qk = _norm_key(question)

    if tech and COL_TECH in df.columns:
        tech_norm = df[COL_TECH].astype(str).map(_norm_tech)
        target = _norm_tech(tech)
        sub = df[tech_norm == target]
        if sub.empty:
            sub = df[tech_norm.str.contains(re.escape(target), case=False, na=False)]
        if sub.empty:
            return f"No customers found with Tech = '{target}'."

        customers = sub[COL_NAME].astype(str).map(_norm_text) if COL_NAME in sub.columns else pd.Series([""] * len(sub))
        uniq = []
        seen = set()
        for c in customers.tolist():
            if not c.strip():
                continue
            ck = _norm_key(c)
            if ck not in seen:
                seen.add(ck)
                uniq.append(c)

        total_balance = sub[COL_BALANCE].astype(str).map(parse_money_to_float).sum() if COL_BALANCE in sub.columns else 0.0
        total_fee = sub[COL_FEE].astype(str).map(parse_money_to_float).sum() if COL_FEE in sub.columns else 0.0
        total_run = sub[COL_TOTAL_RUN].astype(str).map(parse_money_to_float).sum() if COL_TOTAL_RUN in sub.columns else 0.0

        wants_list = any(k in qk for k in ["list", "which customers", "customers"])
        wants_count = any(k in qk for k in ["how many", "count"])
        wants_balance_sum = any(k in qk for k in ["total balance", "total", "sum"])

        if not (wants_list or wants_count or wants_balance_sum):
            wants_list = True
            wants_balance_sum = True

        lines = [
            f"Tech: {target}",
            f"Customers (unique by Name): {len(uniq)}",
            f"Total balance: {total_balance:g}",
        ]

        if "fee" in qk:
            lines.append(f"Total fee: {total_fee:g}")
        if "total customer spend" in qk or "total spend" in qk or "run" in qk:
            lines.append(f"Total customer spend: {total_run:g}")

        if wants_list:
            lines.append("Customer list:")
            for c in uniq[:200]:
                first_row = int(sub.loc[customers.map(_norm_key) == _norm_key(c), "_row_index"].iloc[0]) if COL_NAME in sub.columns else int(sub["_row_index"].iloc[0])
                lines.append(f"- {c} (Row {first_row})")
            if len(uniq) > 200:
                lines.append(f"... (showing 200/{len(uniq)} customers)")
        return "\n".join(lines)

    return "This looks like a summary/statistics question. Please specify a Tech (for example: 'Tech Hoang ...') for an exact full-scan answer."


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

    # 1) Read sheet
    client = connect_gsheet(CREDENTIALS_PATH)
    headers, rows = fetch_sheet_rows(client, SHEET_KEY, WORKSHEET_INDEX)

    print(f"Headers ({len(headers)}): {headers}")
    print(f"Rows: {len(rows)}")
    if not rows:
        print("No data rows found.")
        return

    # 2) Embeddings + vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not os.path.exists(FAISS_DIR):
        docs = build_documents(rows, SHEET_KEY)
        if not docs:
            print("No valid documents to embed.")
            return
        vectorstore = reset_vectorstore_with_docs(embeddings, docs, FAISS_DIR)
        all_hashes = {row_hash(r) for r in rows}
        save_state(STATE_PATH, all_hashes)
        print(f"Created new FAISS index with {len(docs)} docs.")
    else:
        vectorstore = load_or_create_vectorstore(embeddings, FAISS_DIR)
        added = ingest_incremental(rows, vectorstore, embeddings)
        print(f"Incremental ingest: added {added} docs.")

    # 3) Chains
    rag_chain = build_rag_chain(vectorstore)

    print("\n=== Ready ===")
    print("Tips:")
    print("- Stats questions (count/total/list/statistics) => FULL SCAN (exact)")
    print("- Other questions => RAG retrieval")
    print("Type 'reload' to reload the sheet, or 'exit' to quit.")

    while True:
        q = input("\nQ> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "reload":
            headers, rows = fetch_sheet_rows(client, SHEET_KEY, WORKSHEET_INDEX)
            print(f"Reloaded rows: {len(rows)}")
            continue

        answer = full_scan_stats_answer(rows, q) if is_stats_question(q) else rag_chain.invoke(q)
        print("\nANSWER:\n", answer)


if __name__ == "__main__":
    main()
