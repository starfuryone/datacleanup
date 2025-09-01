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
