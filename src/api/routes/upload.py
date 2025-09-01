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
