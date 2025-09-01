from __future__ import annotations
from pathlib import Path
import pandas as pd

class FileReader:
    @staticmethod
    def read_any(path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() in {".csv"}:
            return pd.read_csv(p)
        if p.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(p)
        raise ValueError(f"Unsupported file type: {p.suffix}")
