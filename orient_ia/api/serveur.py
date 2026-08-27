"""Serveur Web FastAPI reliant l'Agent ORIENT'IA et l'Interface utilisateur."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orient_ia.agent.orchestrateur import AgentOrientIA

app = FastAPI(
    title="ORIENT'IA API",
    description="Backend d'aide à l'orientation pédagogique combinant ML, RAG et Agent conversationnel.",
    version="1.0.0",
)

# Configuration CORS pour permettre la communication avec l'interface web locale
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instanciation de l'agent unique traçable
agent = AgentOrientIA()


class ProfilPayload(BaseModel):
    niveau: Optional[str] = None
    serie_bacc: Optional[str] = "C"
    moyenne_generale: Optional[float] = 12.5
    matieres_preferees: Optional[Any] = ""
    competences: Optional[str] = ""
    centres_interet: Optional[Any] = []
    projets: Optional[str] = ""
    preferences_professionnelles: Optional[str] = ""
    environnement_travail: Optional[str] = ""


class MessagePayload(BaseModel):
    message: str = Field(..., description="Message ou question de l'utilisateur")
    profil: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Profil utilisateur courant")


@app.get("/sante")
def verifier_sante():
    """Vérification de l'état de l'API."""
    return {"statut": "operationnel", "service": "ORIENT'IA"}


@app.post("/profil/analyser")
def analyser_profil(payload: ProfilPayload):
    """Analyse un profil candidat via le modèle ML et les règles RAG."""
    profil_dict = payload.model_dump()
    if not payload.niveau:
        return {
            "recommandations": [],
            "sources": [],
            "erreur": "profil_incomplet",
            "message": "Veuillez renseigner au moins votre niveau d'études.",
        }

    resultat = agent.outils.analyser_profil_ml(profil_dict)

    # Construction de la traçabilité pour l'interface
    tracabilite = {
        "question": "Analyse de profil candidat",
        "profil": f"Niveau: {payload.niveau}, Intérêts: {payload.centres_interet}",
        "outils": "analyser_profil_ml (ExtraTrees Classifier) + RAG ISPM",
        "resultats": f"{len(resultat.get('recommandations', []))} piste(s) générée(s) avec niveau d'incertitude modéré.",
    }

    return {
        "recommandations": resultat.get("recommandations", []),
        "sources": resultat.get("sources", []),
        "tracabilite": tracabilite,
        "incertitude_globale": resultat.get("incertitude_globale", "Modérée"),
    }


@app.post("/agent/message")
def envoyer_message(payload: MessagePayload):
    """Interaction conversationnelle sécurisée avec l'Agent ORIENT'IA."""
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")

    reponse = agent.traiter_message(payload.message, payload.profil)
    return reponse


@app.get("/traces")
def get_traces():
    """Observabilité : retourne l'historique complet des traces d'exécution."""
    return {"nb_traces": len(agent.historique_traces), "traces": [asdict(t) for t in agent.historique_traces]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
