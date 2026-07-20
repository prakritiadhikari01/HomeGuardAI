from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.application.runtime_manager import RuntimeManager

runtime_manager = RuntimeManager()


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting HomeGuard AI Engine")
    print("=" * 60)

    runtime_manager.start()

    print("✓ RuntimeManager started")
    print("✓ Camera synchronization started")
    print("=" * 60)

    yield

    print()
    print("=" * 60)
    print("Stopping HomeGuard AI Engine")
    print("=" * 60)

    runtime_manager.stop_all()

    print("✓ Runtime stopped")
    print("=" * 60)


app = FastAPI(
    title="HomeGuard AI Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://192.168.1.14:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "HomeGuard AI Engine",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }