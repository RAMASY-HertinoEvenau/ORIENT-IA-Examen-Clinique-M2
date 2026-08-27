"""Modèles de données utiles au RAG documentaire."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentRAG:
    """Un passage documentaire indexable avec sa provenance."""

    identifiant: str
    titre: str
    type_document: str
    contenu: str
    source: str
    url: str | None = None
    date_consultation: str | None = None
    statut: str | None = None
    identifiant_source: str = ""
    metadonnees: dict[str, Any] = field(default_factory=dict)

    def vers_resultat(self, score: float) -> dict[str, Any]:
        """Convertit le document RAG en résultat exploitable par l'agent."""
        return {
            "contenu": self.contenu,
            "score": float(score),
            "identifiant": self.identifiant,
            "titre": self.titre,
            "type_document": self.type_document,
            "source": self.source,
            "url": self.url,
            "date_consultation": self.date_consultation,
            "statut": self.statut,
            "identifiant_source": self.identifiant_source,
            "metadonnees": dict(self.metadonnees),
        }
