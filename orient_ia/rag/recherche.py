"""API de recherche documentaire locale pour ORIENT'IA."""
from __future__ import annotations

from orient_ia.rag.chargeur import charger_documents_pedagogiques
from orient_ia.rag.index import IndexRechercheDocuments


def rechercher_documents(
    question: str,
    nombre_resultats: int = 5,
    seuil: float = 0.05,
) -> dict[str, object]:
    """Recherche les documents pédagogiques les plus pertinents pour une question."""
    if not question or not question.strip():
        return {
            "trouve": False,
            "resultats": [],
            "message": "Aucune question n'a été fournie pour la recherche documentaire.",
        }

    documents = charger_documents_pedagogiques()
    index = IndexRechercheDocuments(documents)
    resultats = index.rechercher(question, nombre_resultats=nombre_resultats, seuil=seuil)

    if not resultats:
        return {
            "trouve": False,
            "resultats": [],
            "message": "Aucune information suffisamment pertinente n'a été trouvée dans le corpus.",
        }

    return {
        "trouve": True,
        "resultats": resultats,
        "message": "Informations documentaires pertinentes retrouvées dans le corpus.",
    }
