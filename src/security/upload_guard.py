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
