#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating folders"
mkdir -p src/api/routes src/api/dependencies src/core src/worker src/io src/services src/security scripts

write() { # write <path> <<'PY' ... PY
  local p="$1"; shift
  mkdir -p "$(dirname "$p")"
  cat > "$p"
  echo "  + $p"
}

write src/main.py <<'PY'
from fastapi import FastAPI
from src.api.routes import upload as upload_routes
from src.api.routes import jobs as jobs_routes
# from src.api.routes import ai as ai_routes  # enable later when wired

app = FastAPI(title="DataCleanup Pro API", version="0.1.0")

app.include_router(upload_routes.router)
app.include_router(jobs_routes.router)
# app.include_router(ai_routes.router)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
PY

write src/api/routes/upload.py <<'PY'
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from src.api.dependencies import get_current_user, User
from src.security.upload_guard import UploadGuard
from src.io.storage import StorageService
from src.core.jobs import JobManager

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

@router.post("")
async def upload_files(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
):
    guard = UploadGuard()
    storage = StorageService()
    jobman = JobManager()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    stored_paths = []
    for f in files:
        await guard.validate_file(f)
        path = await storage.store_file(f, user_id=user.id)
        stored_paths.append(path)

    job = jobman.create_job(user_id=user.id, source_paths=stored_paths)
    return {"job_id": job["id"], "status": job["status"], "files": stored_paths}
PY

write src/api/routes/jobs.py <<'PY'
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.core.jobs import JobManager

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

@router.post("")
def create_job():
    jm = JobManager()
    job = jm.create_job(user_id="demo-user", source_paths=[])
    return job

@router.get("/{job_id}")
def get_job(job_id: str):
    jm = JobManager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/{job_id}/preview")
def preview_job(job_id: str, limit: int = Query(100, ge=1, le=1000)):
    jm = JobManager()
    preview = jm.get_preview(job_id=job_id, limit=limit)
    if preview is None:
        raise HTTPException(status_code=404, detail="Job not ready or not found")
    return {"job_id": job_id, "preview": preview}

@router.get("/{job_id}/result")
def download_result(job_id: str, format: Optional[str] = "csv"):
    jm = JobManager()
    url = jm.get_result_url(job_id=job_id, format=format or "csv")
    if not url:
        raise HTTPException(status_code=404, detail="Result not available")
    return {"job_id": job_id, "url": url, "format": format or "csv"}
PY

write src/api/dependencies.py <<'PY'
class User:
    def __init__(self, id: str = "demo-user"):
        self.id = id

def get_current_user() -> "User":
    # TODO: Replace with real auth (JWT/session)
    return User()
PY

write src/security/upload_guard.py <<'PY'
from fastapi import UploadFile, HTTPException

class UploadGuard:
    MAX_SIZE_MB = 50
    ALLOWED_CONTENT_TYPES = {
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",  # some browsers send this for csv/xlsx
    }

    async def validate_file(self, file: UploadFile) -> bool:
        if file.content_type not in self.ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
        # TODO: enforce size by streaming/peeking the file; stubbed for now
        return True
PY

write src/core/jobs.py <<'PY'
from enum import Enum
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime, timezone

from src.io.urls import URLBuilder

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"

# In-memory job store (replace with DB/Redis later)
_STORE: Dict[str, Dict[str, Any]] = {}

class JobManager:
    def __init__(self):
        self.urlb = URLBuilder()

    def create_job(self, user_id: str, source_paths: List[str]) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "user_id": user_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_paths": source_paths,
            "result_key": None,
            "preview": [{"row": 1, "sample": {"name": "Alice"}}, {"row": 2, "sample": {"name": "Bob"}}],
        }
        _STORE[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return _STORE.get(job_id)

    def get_preview(self, job_id: str, limit: int = 100):
        job = _STORE.get(job_id)
        if not job:
            return None
        return job.get("preview", [])[:limit]

    def get_result_url(self, job_id: str, format: str = "csv") -> Optional[str]:
        job = _STORE.get(job_id)
        if not job:
            return None
        key = f"results/{job_id}/output.{format}"
        return self.urlb.build_download_url(key)
PY

write src/worker/run_job.py <<'PY'
# Placeholder worker that would load source files, run cleaning pipeline, and store outputs
# Replace with real queue/cron wiring later.
from src.services.cleaning import clean_dataframe
from src.services.formatting import normalize_dataframe
from src.services.dedupe import dedupe_dataframe

def run_job_from_paths(paths):
    # TODO: load files into DataFrames, merge, process, export
    # This is a stub for local testing
    return {"rows_processed": 0, "result_key": None}
PY

write src/io/storage.py <<'PY'
from pathlib import Path
from typing import Optional
from fastapi import UploadFile

class StorageService:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or "./.uploads")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def store_file(self, file: UploadFile, user_id: str) -> str:
        user_dir = self.base_dir / user_id
        user_dir.mkdir(exist_ok=True, parents=True)
        dest = user_dir / file.filename
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return str(dest)

    def get_signed_url(self, key: str, expires: int = 3600) -> str:
        # Stubbed: in production use boto3.generate_presigned_url for S3/MinIO
        return f"/download/{key}?expires={expires}"
PY

write src/io/urls.py <<'PY'
import os

class URLBuilder:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:8000")

    def build_download_url(self, key: str) -> str:
        return f"{self.base_url}/files/{key}"
PY

write src/services/cleaning.py <<'PY'
import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleanup: trim strings, drop all-empty rows
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    df = df.dropna(how="all")
    return df
PY

write src/services/formatting.py <<'PY'
import pandas as pd

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Example normalization: title-case name columns
    for col in df.columns:
        if "name" in col.lower() and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.title()
    return df
PY

write src/services/merge.py <<'PY'
import pandas as pd
from typing import List

def merge_dataframes(dfs: List[pd.DataFrame], on: str | None = None) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    # Simple concat; replace with smarter join logic later
    return pd.concat(dfs, ignore_index=True)
PY

write src/services/dedupe.py <<'PY'
import pandas as pd

def dedupe_dataframe(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep)
PY

write src/services/export_excel.py <<'PY'
import pandas as pd

def export_excel(df: pd.DataFrame, path: str) -> str:
    df.to_excel(path, index=False)
    return path
PY

write src/services/pdf_report.py <<'PY'
def generate_pdf_report(path: str, metrics: dict) -> str:
    # TODO: implement with reportlab or weasyprint
    with open(path, "w") as f:
        f.write("PDF REPORT\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    return path
PY

write src/services/diff.py <<'PY'
import pandas as pd

def preview_diff(before: pd.DataFrame, after: pd.DataFrame, limit: int = 100) -> dict:
    added = len(after) - len(before)
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_added": max(0, added),
        "sample_after": after.head(limit).to_dict(orient="records"),
    }
PY

write src/services/colmap.py <<'PY'
from typing import Dict, List

CANON = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "tel", "telephone"],
    "name": ["name", "full_name", "fullname"],
}

def heuristic_map(columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for c in columns:
        lc = c.lower().strip()
        for canon, aliases in CANON.items():
            if lc == canon or lc in aliases:
                mapping[c] = canon
                break
    return mapping
PY

write src/validation.py <<'PY'
import pandas as pd

def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> dict:
    required = required or []
    missing = [c for c in required if c not in df.columns]
    return {"ok": not missing, "missing": missing}
PY

write scripts/run_worker.py <<'PY'
#!/usr/bin/env python3
# Simulated worker entrypoint
from src.worker.run_job import run_job_from_paths

if __name__ == "__main__":
    result = run_job_from_paths([])
    print("Worker finished:", result)
PY
chmod +x scripts/run_worker.py

write .env.example <<'ENV'
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET=datacleanup-uploads
BASE_URL=http://localhost:8000
OPENAI_API_KEY=
ENV

write requirements.txt <<'REQ'
fastapi
uvicorn[standard]
pandas
openpyxl
reportlab
boto3
python-multipart
fuzzywuzzy
python-Levenshtein
REQ

echo "==> Done. Files written."
