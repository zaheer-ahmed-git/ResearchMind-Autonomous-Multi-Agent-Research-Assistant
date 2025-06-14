"""FastAPI application factory for the Autonomous Research Agent."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_agent import __version__
from research_agent.api.routes import router
from research_agent.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Autonomous Research Agent",
        description=(
            "LangGraph research agent with task decomposition, memory pruning, "
            "adaptive ranking, recursive refinement, and execution-trace APIs."
        ),
        version=__version__,
    )
    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
