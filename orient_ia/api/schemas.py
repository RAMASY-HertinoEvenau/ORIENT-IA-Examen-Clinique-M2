from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfilRecommandation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    identifiant: str | None = None
    matieres_preferees: list[str] | None = None
    moyenne_scolaire: float | None = Field(default=None, ge=0, le=20)
    competences: dict[str, int] | None = None
    centres_interet: list[str] | None = None
    projets: list[str] | None = None
    preferences_professionnelles: str | None = None
    environnement_travail: str | None = None

    def to_profil_payload(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        return data


class RecommandationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profil: ProfilRecommandation


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    profil: ProfilRecommandation | None = None
    session_id: str | None = None
