import pandas as pd

def dedupe_dataframe(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep)
