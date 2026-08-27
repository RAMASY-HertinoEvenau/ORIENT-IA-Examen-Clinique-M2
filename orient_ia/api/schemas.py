from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfilRecommandation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    identifiant: str | None = None
    matieres_preferees: list[str] | str | None = None
    moyenne_scolaire: float | None = Field(default=None, ge=0, le=20)
    competences: dict[str, int] | list[str] | str | None = None
    centres_interet: list[str] | str | None = None
    projets: list[str] | str | None = None
    preferences_professionnelles: str | None = None
    environnement_travail: str | None = None

    def to_profil_payload(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        if isinstance(data.get("matieres_preferees"), str):
            data["matieres_preferees"] = [m.strip() for m in data["matieres_preferees"].split(",") if m.strip()]
        if isinstance(data.get("competences"), (str, list)):
            data["competences"] = {
                "competence-techniques-informatiques-gestion": 4,
                "competence-electronique-systemes": 3,
            }
        return data


class RecommandationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profil: ProfilRecommandation


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    profil: ProfilRecommandation | None = None
    session_id: str | None = None
