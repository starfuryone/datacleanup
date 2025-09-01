import pandas as pd

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Example normalization: title-case name columns
    for col in df.columns:
        if "name" in col.lower() and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.title()
    return df
