import pandas as pd

def preview_diff(before: pd.DataFrame, after: pd.DataFrame, limit: int = 100) -> dict:
    added = len(after) - len(before)
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_added": max(0, added),
        "sample_after": after.head(limit).to_dict(orient="records"),
    }
