from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from orient_ia.moteur_recommandation import MoteurRecommandation


class RecommendationService:
    _instance: RecommendationService | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, moteur: MoteurRecommandation | None = None) -> None:
        if getattr(self, "_initialise", False):
            return
        self.moteur = moteur or MoteurRecommandation()
        self._initialise = True
        self._original_recommander = self.recommander

    def recommander(self, profil: dict[str, Any]) -> dict[str, Any]:
        try:
            resultat = self.moteur.recommander(profil)
        except Exception as exc:  # pragma: no cover - exercised through HTTP layer
            raise HTTPException(
                status_code=500,
                detail=f"Erreur interne du service de recommandation: {exc}",
            ) from exc

        if not isinstance(resultat, dict):
            raise HTTPException(
                status_code=500,
                detail="Le moteur de recommandation n'a pas produit un résultat exploitable.",
            )

        if resultat.get("status") == "incomplet":
            raise HTTPException(
                status_code=400,
                detail=resultat.get("message", "Profil incomplet : données insuffisantes."),
            )

        recommandations = resultat.get("recommandations", [])
        sources = []
        for recommandation in recommandations:
            for source in recommandation.get("source_documentaire", []):
                if source not in sources:
                    sources.append(source)

        avertissements = [
            "La recommandation est indicative et ne constitue pas une décision officielle d'admission.",
            resultat.get("message", "Recommandation produite à partir du modèle final gelé."),
            resultat.get("incertitude", "Le système demeure prudent et indicatif."),
        ]
        for recommandation in recommandations:
            for limite in recommandation.get("limites", []):
                if limite not in avertissements:
                    avertissements.append(limite)

        return {
            "status": resultat.get("status", "ok"),
            "message": resultat.get("message", "Recommandation produite à partir du modèle final gelé."),
            "recommandations": recommandations,
            "incertitude": resultat.get("incertitude", "Le système demeure prudent et indicatif."),
            "profil": resultat.get("profil", profil),
            "avertissements": avertissements,
            "sources": sources,
        }
