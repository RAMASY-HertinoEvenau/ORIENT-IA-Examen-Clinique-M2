from __future__ import annotations

import inspect
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from orient_ia.agent.orchestrateur import traiter_message
from orient_ia.api.schemas import AgentChatRequest, RecommandationRequest
from orient_ia.api.service import RecommendationService


def _appeler_service(service_instance: RecommendationService, payload: dict[str, object]) -> dict[str, object]:
    methode = service_instance.recommander
    original = getattr(service_instance, "_original_recommander", None)
    try:
        try:
            signature = inspect.signature(methode)
        except (TypeError, ValueError):
            return methode(payload)

        params_pos = [
            param
            for param in signature.parameters.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        if not params_pos:
            return methode()
        return methode(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du service de recommandation: {exc}",
        ) from exc
    finally:
        if original is not None:
            service_instance.recommander = original


def create_app(service: RecommendationService | None = None) -> FastAPI:
    app = FastAPI(title="ORIENT'IA", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service_instance = service or RecommendationService()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/recommandation")
    async def recommander(req: RecommandationRequest) -> dict[str, object]:
        return _appeler_service(service_instance, req.profil.to_profil_payload())

    @app.post("/api/agent/chat")
    async def agent_chat(req: AgentChatRequest) -> dict[str, object]:
        profil_payload = req.profil.to_profil_payload() if req.profil is not None else None
        return traiter_message(
            message=req.message,
            profil=profil_payload,
            session_id=req.session_id,
        )

    dossier_interface = Path(__file__).resolve().parents[2] / "interface"
    if dossier_interface.exists():
        app.mount("/", StaticFiles(directory=str(dossier_interface), html=True), name="interface")

    return app


app = create_app()
