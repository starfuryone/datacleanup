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
