from fastapi import FastAPI

from auth_service.database import close_db, get_client, init_db
from auth_service.routes.auth_routes import router as auth_router


app = FastAPI(title="FastAPI JWT Auth", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    get_client().admin.command("ping")
    init_db()


@app.on_event("shutdown")
def on_shutdown() -> None:
    close_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(auth_router)
