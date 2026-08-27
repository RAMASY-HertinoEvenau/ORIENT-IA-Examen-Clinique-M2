"""Modèles de données pour l'agent conversationnel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DemandeAgent:
    message: str
    session_id: str | None = None
    profil: dict[str, Any] | None = None


@dataclass(slots=True)
class TraceConversation:
    session_id: str | None
    message: str
    etat: str
    outils_appeles: list[str] = field(default_factory=list)
    reponse: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    donnees_profil: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "message": self.message,
            "etat": self.etat,
            "outils_appeles": self.outils_appeles,
            "reponse": self.reponse,
            "sources": self.sources,
            "donnees_profil": self.donnees_profil,
        }
