import pandas as pd
from typing import List

def merge_dataframes(dfs: List[pd.DataFrame], on: str | None = None) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    # Simple concat; replace with smarter join logic later
    return pd.concat(dfs, ignore_index=True)
