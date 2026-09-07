from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .routers import accidents, admin, casualties, predict, vehicles  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    predict.load_model()
    yield


app = FastAPI(
    title="Traffic Accident Severity Analytics API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(accidents.router)
app.include_router(casualties.router)
app.include_router(vehicles.router)
app.include_router(admin.router)
app.include_router(predict.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
