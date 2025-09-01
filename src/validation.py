import pandas as pd

def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> dict:
    required = required or []
    missing = [c for c in required if c not in df.columns]
    return {"ok": not missing, "missing": missing}
