from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.I)

@dataclass
class ValidationResult:
    df: pd.DataFrame
    counts: Dict[str, int]
    samples: Dict[str, List[int]]

def _flag(df: pd.DataFrame, col: str, mask: pd.Series, label: str, samples: Dict[str, List[int]]):
    if label not in df.columns:
        df[label] = False
    df.loc[mask, label] = True
    if label not in samples:
        # store sample row indices (up to 20 to keep JSON small)
        samples[label] = df.index[mask].tolist()[:20]

def _ensure_status_col(df: pd.DataFrame):
    if "validation_status" not in df.columns:
        df["validation_status"] = ""
    return df

def _append_status(df: pd.DataFrame, label: str):
    mask = df[label] == True  # noqa: E712
    # empty -> label
    df.loc[mask & (df["validation_status"] == ""), "validation_status"] = label
    # append with comma for non-empty
    df.loc[mask & (df["validation_status"] != "") & (df["validation_status"] != label), "validation_status"] = (
        df.loc[mask & (df["validation_status"] != "") & (df["validation_status"] != label), "validation_status"] + "," + label
    )

def validate_dataframe(
    df_in: pd.DataFrame,
    *,
    email_col: str | None = None,
    phone_col: str | None = None,
    date_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, List[int]]]:
    """
    Returns (df_with_status, counts, sample_row_ids_by_flag)

    Rules:
      - Email: invalid if empty or RFC-ish regex fails
      - Phone: invalid/missing if empty after normalization (assumes prior normalization)
      - Dates: invalid if unparsable to ISO; out-of-range if < 1970-01-01 or > today + 1y
      - Cross-company phone duplicates: same phone under >1 company
      - Numeric outliers: for cols named 'amount' or 'price' using 1.5×IQR
    """
    df = df_in.copy()
    df = _ensure_status_col(df)
    counts: Dict[str, int] = {}
    samples: Dict[str, List[int]] = {}

    # Heuristics to find columns
    cols_lower = {c.lower(): c for c in df.columns}
    email_col = email_col or cols_lower.get("email")
    phone_col = phone_col or cols_lower.get("phone")
    company_col = cols_lower.get("company")
    if date_cols is None:
        date_cols = [c for c in df.columns if "date" in c.lower()]

    # 1) Email invalid
    if email_col and email_col in df.columns:
        series = df[email_col].astype(str).str.strip()
        mask_invalid_email = series.eq("") | series.isna() | ~series.str.match(EMAIL_RE)
        _flag(df, email_col, mask_invalid_email, "invalid_email", samples)

    # 2) Phone missing/invalid (post-normalization expected)
    if phone_col and phone_col in df.columns:
        series = df[phone_col].astype(str).str.strip()
        mask_missing_phone = series.eq("") | series.isna()
        _flag(df, phone_col, mask_missing_phone, "missing_phone", samples)

    # 3) Date columns: invalid or out-of-range
    lower_bound = pd.Timestamp("1970-01-01")
    upper_bound = pd.Timestamp(datetime.utcnow() + timedelta(days=365))
    for dc in date_cols:
        if dc not in df.columns:
            continue
        parsed = pd.to_datetime(df[dc], errors="coerce", utc=False)
        mask_invalid_date = parsed.isna() & df[dc].notna()
        mask_oor = parsed.notna() & ((parsed < lower_bound) | (parsed > upper_bound))
        _flag(df, dc, mask_invalid_date, "invalid_date", samples)
        _flag(df, dc, mask_oor, "date_out_of_range", samples)

    # 4) Cross-company phone duplicates
    if phone_col and company_col and all(c in df.columns for c in (phone_col, company_col)):
        tmp = df[[phone_col, company_col]].fillna("").astype(str)
        # Phones that appear under >1 distinct company (exclude empty phones)
        multi_company = (
            tmp[tmp[phone_col] != ""]
            .groupby(phone_col)[company_col]
            .nunique()
            .rename("n_companies")
        )
        bad_phones = set(multi_company[multi_company > 1].index)
        mask_cross = df[phone_col].astype(str).isin(bad_phones)
        _flag(df, phone_col, mask_cross, "cross_company_phone_duplicate", samples)

    # 5) Numeric outliers (amount/price)
    for cn in df.columns:
        if cn.lower() in {"amount", "price"}:
            s = pd.to_numeric(df[cn], errors="coerce")
            if s.notna().sum() == 0:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask_out = s.notna() & ((s < lo) | (s > hi))
            _flag(df, cn, mask_out, "numeric_outlier", samples)

    # Build counts & status strings
    labels = [c for c in df.columns if c in {
        "invalid_email", "missing_phone", "invalid_date", "date_out_of_range",
        "cross_company_phone_duplicate", "numeric_outlier"
    }]
    for lb in labels:
        _append_status(df, lb)
        counts[lb] = int(df[lb].sum())

    return df, counts, samples
