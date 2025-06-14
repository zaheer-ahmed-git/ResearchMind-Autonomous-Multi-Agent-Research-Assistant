from fastapi import APIRouter
from research_agent import __version__
from research_agent.config import get_settings

router = APIRouter()

@router.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "version": __version__, "use_llm": settings.use_llm and bool(settings.openai_api_key)}
