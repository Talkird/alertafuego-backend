import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import account, alerts, detect, detections, reports, stats, telegram

# The model and Earth Engine load on the first detection run (detect.get_inference_ctx),
# not at startup: the endpoints the app calls don't need them, and loading them made
# every cold start slow.
app = FastAPI()

load_dotenv()
frontend_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(detections.router)
app.include_router(reports.router)
app.include_router(alerts.router)
app.include_router(stats.router)
app.include_router(account.router)
app.include_router(telegram.router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
