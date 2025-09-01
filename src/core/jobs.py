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
