import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleanup: trim strings, drop all-empty rows
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    df = df.dropna(how="all")
    return df
