#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheet Assistant v3
- Mode 1 (Column search / per-customer): e.g. "What is the balance of customer C305?"
- Mode 2 (Full-table stats): e.g. "Which customers are managed by Tech Hoang? Total balance"

This script reads a Google Sheet and answers questions with deterministic DataFrame logic
(for counting/summing/listing), avoiding RAG retrieval misses.
"""

import os
import re
from typing import List, Optional
from dotenv import load_dotenv

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

load_dotenv()

# =========================
# CONFIG
# =========================
SHEET_KEY = os.getenv("SHEET_KEY")
WORKSHEET_INDEX = 0
CREDENTIALS_PATH = "config/credentials.json"

# Normalization / alias for Tech names
TECH_ALIAS = {
    "hoang": "Hoang",
    "hai": "Hai",
    "an": "An",
    "trung": "Trung",
}

# Core columns
COL_NAME = "Name"
COL_TECH = "Tech"
COL_ACCOUNT = "Account"
COL_FEE = "Fee"
COL_BALANCE = "Balance"
COL_TOTAL_RUN = "Total Customer Spend"
MAX_LIST_ROWS = 200

AUTO_COL_PREFIX = "__col_"


# =========================
# GOOGLE SHEET IO
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


def fetch_sheet_df(client: gspread.Client, sheet_key: str, worksheet_index: int = 0) -> pd.DataFrame:
    sh = client.open_by_key(sheet_key)
    ws = sh.get_worksheet(worksheet_index)
    values = ws.get_all_values()
    if not values or len(values) < 1:
        return pd.DataFrame()

    headers = normalize_headers(values[0])
    data_rows = values[1:]

    cleaned = []
    for r in data_rows:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        if not any((c or "").strip() for c in r):
            continue
        cleaned.append(r)

    df = pd.DataFrame(cleaned, columns=headers)
    df["_row_index"] = range(2, 2 + len(df))

    # Backward compatibility for existing VN header
    if "Tong Khach chay" in df.columns and COL_TOTAL_RUN not in df.columns:
        df.rename(columns={"Tong Khach chay": COL_TOTAL_RUN}, inplace=True)
    if "BALANCE" in df.columns and COL_BALANCE not in df.columns:
        df.rename(columns={"BALANCE": COL_BALANCE}, inplace=True)

    return df


# =========================
# NORMALIZATION HELPERS
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


# =========================
# QUESTION PARSING
# =========================
FIELD_SYNONYMS = {
    "Balance": [r"\bbalance\b"],
    "Fee": [r"\bfee\b", r"\bservice\s*fee\b"],
    COL_TOTAL_RUN: [r"\btotal\s*customer\s*spend\b", r"\btotal\s*spend\b", r"\brun\b"],
    "Tech": [r"\btech\b", r"\btechnical\b", r"\bowner\b", r"\bmanaged\s*by\b"],
    "Account": [r"\baccount\b", r"\baccountant\b"],
    "Name": [r"\bcustomer\b", r"\bname\b", r"\bcode\b"],
}


def detect_requested_field(question: str) -> Optional[str]:
    q = _norm_key(question)
    for field, patterns in FIELD_SYNONYMS.items():
        for p in patterns:
            if re.search(p, q, flags=re.IGNORECASE):
                return field
    return None


def extract_codes(question: str) -> List[str]:
    q = _norm_text(question)
    found = []
    for m in re.finditer(r"\b([cCpP])\s*[-_]?\s*(\d{1,6})\b", q):
        found.append(f"{m.group(1).upper()}{m.group(2)}")

    out = []
    seen = set()
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def detect_stats_intent(question: str) -> bool:
    q = _norm_key(question)
    return any(k in q for k in ["how many", "total", "statistics", "list", "count", "sum"])


def extract_tech_name(question: str) -> Optional[str]:
    q = _norm_text(question)

    m = re.search(r"(?:tech|technical)\s*[:=\-]?\s*([A-Za-z\s\.]+)", q, flags=re.IGNORECASE)
    if m:
        name = " ".join(m.group(1).strip().split()[:3])
        return _norm_tech(name)

    m = re.search(r"([A-Za-z\s\.]{2,30})\s+manages", q, flags=re.IGNORECASE)
    if m:
        name = " ".join(m.group(1).strip().split()[-3:])
        return _norm_tech(name)

    return None


# =========================
# CORE LOGIC
# =========================
def filter_df_contains_all(df: pd.DataFrame, tokens: List[str], preferred_col: Optional[str] = None) -> pd.DataFrame:
    if df.empty or not tokens:
        return df.iloc[0:0]

    def _contains(series: pd.Series, token: str) -> pd.Series:
        return series.astype(str).str.contains(re.escape(token), case=False, na=False)

    if preferred_col and preferred_col in df.columns:
        mask = pd.Series([True] * len(df))
        for t in tokens:
            mask &= _contains(df[preferred_col], t)
        sub = df[mask]
        if not sub.empty:
            return sub

    mask = pd.Series([True] * len(df))
    for t in tokens:
        token_mask = pd.Series([False] * len(df))
        for c in df.columns:
            if c == "_row_index":
                continue
            token_mask |= _contains(df[c], t)
        mask &= token_mask
    return df[mask]


def answer_per_customer(df: pd.DataFrame, question: str) -> str:
    """Mode 1: per-customer lookup."""
    tokens = extract_codes(question)
    requested_field = detect_requested_field(question) or COL_BALANCE

    preferred = COL_NAME if COL_NAME in df.columns else None
    matched = filter_df_contains_all(df, tokens, preferred_col=preferred) if tokens else df.iloc[0:0]

    if matched.empty and tokens:
        return f"No customer found for code: {', '.join(tokens)}."
    if matched.empty:
        return "I do not know (please provide a customer code like C305 or a clearer filter)."

    col = requested_field if requested_field in df.columns else None
    if col is None:
        fallback_map = {
            "Balance": COL_BALANCE,
            "Fee": COL_FEE,
            COL_TOTAL_RUN: COL_TOTAL_RUN,
            "Tech": COL_TECH,
            "Account": COL_ACCOUNT,
            "Name": COL_NAME,
        }
        mapped = fallback_map.get(requested_field)
        if mapped in df.columns:
            col = mapped

    lines = []
    if len(matched) == 1:
        r = matched.iloc[0]
        row_idx = int(r["_row_index"])
        name = _norm_text(r.get(COL_NAME, ""))
        tech = _norm_tech(r.get(COL_TECH, "")) if COL_TECH in matched.columns else ""
        acct = _norm_text(r.get(COL_ACCOUNT, "")) if COL_ACCOUNT in matched.columns else ""
        val = _norm_text(r.get(col, "")) if col else ""
        label = col if col else requested_field

        lines.append(f"Customer: {name or '(empty)'} | Tech: {tech or '(empty)'} | Account: {acct or '(empty)'}")
        lines.append(f"{label}: {val if val != '' else '(empty)'} (Row {row_idx})")
        return "\n".join(lines)

    lines.append(f"Found {len(matched)} matching rows ({', '.join(tokens)}).")
    show = matched.head(MAX_LIST_ROWS)
    for _, r in show.iterrows():
        row_idx = int(r["_row_index"])
        name = _norm_text(r.get(COL_NAME, ""))
        tech = _norm_tech(r.get(COL_TECH, "")) if COL_TECH in matched.columns else ""
        val = _norm_text(r.get(col, "")) if col else ""
        if col in (COL_BALANCE, COL_FEE, COL_TOTAL_RUN):
            num = parse_money_to_float(val)
            val_disp = f"{val} (~{num:g})" if val != "" else "(empty)"
        else:
            val_disp = val if val != "" else "(empty)"

        lines.append(
            f"- {name or '(empty)'} | Tech: {tech or '(empty)'} | "
            f"{col or requested_field}: {val_disp} (Row {row_idx})"
        )

    if len(matched) > len(show):
        lines.append(f"... (showing {len(show)}/{len(matched)} rows)")
    return "\n".join(lines)


def answer_stats_full_table(df: pd.DataFrame, question: str) -> str:
    """Mode 2: full-table statistics, primarily by Tech."""
    tech_name = extract_tech_name(question)
    if not tech_name:
        return "I do not know (please specify a Tech, for example: 'Tech Hoang ...')."

    if COL_TECH not in df.columns:
        return f"Column '{COL_TECH}' not found in the sheet."

    tech_series_norm = df[COL_TECH].astype(str).map(_norm_tech)
    target = _norm_tech(tech_name)

    matched = df[tech_series_norm == target]
    if matched.empty:
        matched = df[tech_series_norm.str.contains(re.escape(target), case=False, na=False)]

    if matched.empty:
        return f"No customers found with Tech = '{target}'."

    customers = matched[COL_NAME].astype(str).map(_norm_text) if COL_NAME in matched.columns else pd.Series([""] * len(matched))
    total_balance = matched[COL_BALANCE].astype(str).map(parse_money_to_float).sum() if COL_BALANCE in matched.columns else 0.0

    unique_set = []
    seen = set()
    for c in [c for c in customers.tolist() if c.strip() != ""]:
        k = _norm_key(c)
        if k not in seen:
            seen.add(k)
            unique_set.append(c)

    rows = matched["_row_index"].astype(int).tolist()

    q = _norm_key(question)
    wants_list = any(k in q for k in ["list", "which customers", "customers"])
    wants_total = any(k in q for k in ["total balance", "total", "sum"])

    if not (wants_list or wants_total):
        wants_list = True
        wants_total = True

    lines = [
        f"Tech: {target}",
        f"Matched rows: {len(matched)} (Rows {', '.join(map(str, rows[:50]))}{'...' if len(rows) > 50 else ''})",
        f"Customers (unique by Name): {len(unique_set)}",
    ]

    if wants_total:
        lines.append(f"Total balance: {total_balance:g}")

    if wants_list:
        lines.append("Customer list:")
        show = unique_set[:MAX_LIST_ROWS]
        for c in show:
            first_row = int(matched.loc[customers.map(_norm_key) == _norm_key(c), "_row_index"].iloc[0]) if COL_NAME in matched.columns else rows[0]
            lines.append(f"- {c} (Row {first_row})")
        if len(unique_set) > len(show):
            lines.append(f"... (showing {len(show)}/{len(unique_set)} customers)")

    return "\n".join(lines)


def route_and_answer(df: pd.DataFrame, question: str) -> str:
    """
    Router:
    - If question is stats by Tech => Mode 2
    - Else if it contains customer code like C305 => Mode 1
    - Else if it looks like stats intent => ask for Tech
    - Else => Mode 1 requires code
    """
    tokens = extract_codes(question)
    tech_name = extract_tech_name(question)
    stats_intent = detect_stats_intent(question)

    if tech_name and stats_intent:
        return answer_stats_full_table(df, question)

    if tokens:
        return answer_per_customer(df, question)

    if stats_intent:
        if tech_name:
            return answer_stats_full_table(df, question)
        return "This looks like a statistics question. Please specify a Tech (for example: 'Tech Hoang ...')."

    return "I do not know (ask like: 'Balance of customer C305 ...' or 'Tech Hoang ...')."


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(f"Missing {CREDENTIALS_PATH}")

    client = connect_gsheet(CREDENTIALS_PATH)
    df = fetch_sheet_df(client, SHEET_KEY, WORKSHEET_INDEX)

    print(f"Loaded rows: {len(df)}")
    print("Ready. Examples:")
    print(" - What is the balance of customer C305?")
    print(" - Customer report for C548 P14")
    print(" - Which customers are managed by Tech Hoang? Total balance")
    print("Type 'reload' to reload sheet, or 'exit' to quit.")

    while True:
        q = input("\nQ> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "reload":
            df = fetch_sheet_df(client, SHEET_KEY, WORKSHEET_INDEX)
            print(f"Reloaded rows: {len(df)}")
            continue

        ans = route_and_answer(df, q)
        print("\nANSWER:\n" + ans)


if __name__ == "__main__":
    main()
