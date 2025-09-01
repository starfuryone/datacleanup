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
