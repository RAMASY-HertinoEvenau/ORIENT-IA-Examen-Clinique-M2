"""Chargement du corpus pédagogique versionné et de sa provenance."""

import json
from datetime import date
from pathlib import Path
from typing import Any

from orient_ia.domaine.modeles import (
    Competence,
    CorpusPedagogique,
    Formation,
    Matiere,
    Mention,
    Metier,
    Parcours,
    Passerelle,
    Prerequis,
    Source,
    StatutSource,
)


def charger_corpus(chemin: Path) -> CorpusPedagogique:
    """Charge un corpus JSON et exige la provenance de chaque entité."""
    donnees: dict[str, Any] = json.loads(chemin.read_text(encoding="utf-8"))
    sources = tuple(
        Source(
            titre=source["titre"],
            origine=source["origine"],
            url=source.get("url"),
            date_consultation=date.fromisoformat(source["date_consultation"]),
            statut=StatutSource(source["statut"]),
            donnees_extraites=source["donnees_extraites"],
            limites=source["limites"],
            incertitudes=source["incertitudes"],
            identifiant=source["identifiant"],
            section=source["section"],
            extrait=source["extrait"],
        )
        for source in donnees["sources"]
    )
    corpus = CorpusPedagogique(
        version=donnees["version"],
        sources=sources,
        mentions=tuple(Mention(**item) for item in donnees["mentions"]),
        formations=tuple(Formation(**item) for item in donnees["formations"]),
        parcours=tuple(Parcours(**item) for item in donnees["parcours"]),
        matieres=tuple(Matiere(**item) for item in donnees["matieres"]),
        competences=tuple(Competence(**item) for item in donnees["competences"]),
        prerequis=tuple(Prerequis(**item) for item in donnees["prerequis"]),
        metiers=tuple(Metier(**item) for item in donnees["metiers"]),
        passerelles=tuple(Passerelle(**item) for item in donnees["passerelles"]),
    )
    corpus.valider(exiger_provenance=True)
    return corpus