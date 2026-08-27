"""Service de haut niveau pour le RAG documentaire."""
from __future__ import annotations

from typing import Any

from orient_ia.rag.recherche import rechercher_documents


def rechercher_contexte(
    question: str,
    nombre_resultats: int = 5,
    seuil: float = 0.05,
) -> dict[str, Any]:
    """Retourne un contexte directement exploitable par un agent LLM."""
    resultat = rechercher_documents(question, nombre_resultats=nombre_resultats, seuil=seuil)
    if not resultat["trouve"]:
        return {
            "trouve": False,
            "resultats": [],
            "contexte": "",
            "message": resultat["message"],
        }

    blocs: list[str] = []
    for document in resultat["resultats"]:
        provenance = f"[{document['source']}]"
        if document.get("url"):
            provenance += f" ({document['url']})"
        blocs.append(f"{provenance}\n{document['contenu']}")

    return {
        "trouve": True,
        "resultats": resultat["resultats"],
        "contexte": "\n\n".join(blocs),
        "message": "Contexte documentaire récupéré avec provenance explicite.",
    }
