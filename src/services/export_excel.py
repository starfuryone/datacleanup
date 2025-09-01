import pandas as pd

def export_excel(df: pd.DataFrame, path: str) -> str:
    df.to_excel(path, index=False)
    return path
