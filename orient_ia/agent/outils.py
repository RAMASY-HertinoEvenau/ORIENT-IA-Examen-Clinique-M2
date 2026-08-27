"""Outils concrets de l'agent ORIENT'IA."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from orient_ia.domaine.corpus import charger_corpus
from orient_ia.moteur_recommandation import MoteurRecommandation
from orient_ia.rag.recherche import rechercher_documents


def rechercher_formations(question: str, nombre_resultats: int = 5, seuil: float = 0.05) -> dict[str, Any]:
    """Recherche documentaire locale dans le corpus pédagogique."""
    return rechercher_documents(question=question, nombre_resultats=nombre_resultats, seuil=seuil)


def analyser_profil(profil: Mapping[str, Any] | None) -> dict[str, Any]:
    """Exécute l'inférence du moteur ML à partir d'un profil candidat compatible."""
    if not profil:
        return {"statut": "incomplet", "message": "Profil insuffisant pour une recommandation fiable."}

    moteur = MoteurRecommandation()
    try:
        resultat = moteur.recommander(profil)
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - dépend de l'environnement ML et du profil
        return {"statut": "erreur", "message": f"Impossible de calculer une recommandation: {exc}"}

    return {
        "statut": "ok",
        "message": "Recommandation calculée à partir du profil fourni.",
        "resultat": resultat,
    }


def comparer_parcours(parcours: Iterable[str] | str | None) -> dict[str, Any]:
    """Compare plusieurs parcours de formation à partir du corpus pédagogique."""
    if parcours is None:
        return {"statut": "incomplet", "message": "Aucun parcours fourni pour la comparaison."}

    if isinstance(parcours, str):
        parcours_liste = [p.strip() for p in parcours.split(",") if p.strip()]
    else:
        parcours_liste = [str(item).strip() for item in parcours if str(item).strip()]

    if not parcours_liste:
        return {"statut": "incomplet", "message": "Aucune donnée de parcours n'est exploitable."}

    chemin_corpus = Path(__file__).resolve().parents[2] / "donnees" / "corpus_pedagogique.json"
    corpus = charger_corpus(chemin_corpus)
    comparatif = []
    for item in parcours_liste:
        for parcours_corpus in corpus.parcours:
            if item.lower() in parcours_corpus.nom.lower() or item.lower() == parcours_corpus.identifiant.lower():
                comparatif.append(
                    {
                        "identifiant": parcours_corpus.identifiant,
                        "nom": parcours_corpus.nom,
                        "matieres": list(parcours_corpus.matieres),
                        "competences": list(parcours_corpus.competences),
                        "metiers": list(parcours_corpus.metiers),
                    }
                )
                break

    if not comparatif:
        return {"statut": "not_found", "message": "Aucun parcours correspondant n'a été trouvé dans le corpus.", "parcours": parcours_liste}

    return {"statut": "ok", "message": "Comparaison des parcours effectuée.", "resultat": comparatif}
