import os

class URLBuilder:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:8000")

    def build_download_url(self, key: str) -> str:
        return f"{self.base_url}/files/{key}"
