from __future__ import annotations
from pathlib import Path
import zipfile
import json
import pandas as pd

class FileWriter:
    @staticmethod
    def write_csv(df: pd.DataFrame, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def write_json(obj: dict, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def bundle_zip(files: dict[str, str | Path], zip_path: str | Path):
        Path(zip_path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arcname, fpath in files.items():
                zf.write(fpath, arcname=arcname)
