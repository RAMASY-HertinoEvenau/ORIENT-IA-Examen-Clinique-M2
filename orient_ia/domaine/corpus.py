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
        mentions=tuple(Mention(identifiant=item["identifiant"], nom=item["nom"], parcours=tuple(item.get("parcours", ())), sources=tuple(item.get("sources", ()))) for item in donnees["mentions"]),
        formations=tuple(Formation(identifiant=item["identifiant"], nom=item["nom"], mentions=tuple(item.get("mentions", ())), niveau=item.get("niveau", ""), sources=tuple(item.get("sources", ()))) for item in donnees["formations"]),
        parcours=tuple(Parcours(identifiant=item["identifiant"], nom=item["nom"], matieres=tuple(item.get("matieres", ())), competences=tuple(item.get("competences", ())), prerequis=tuple(item.get("prerequis", ())), metiers=tuple(item.get("metiers", ())), sources=tuple(item.get("sources", ()))) for item in donnees["parcours"]),
        matieres=tuple(Matiere(identifiant=item["identifiant"], nom=item["nom"], sources=tuple(item.get("sources", ()))) for item in donnees["matieres"]),
        competences=tuple(Competence(identifiant=item["identifiant"], nom=item["nom"], description=item.get("description", ""), sources=tuple(item.get("sources", ()))) for item in donnees["competences"]),
        prerequis=tuple(Prerequis(identifiant=item["identifiant"], description=item["description"], obligatoire=item.get("obligatoire", True), sources=tuple(item.get("sources", ()))) for item in donnees["prerequis"]),
        metiers=tuple(Metier(identifiant=item["identifiant"], nom=item["nom"], description=item.get("description", ""), sources=tuple(item.get("sources", ()))) for item in donnees["metiers"]),
        passerelles=tuple(Passerelle(identifiant=item["identifiant"], source=item["source"], cible=item["cible"], description=item.get("description", ""), sources=tuple(item.get("sources", ()))) for item in donnees["passerelles"]),
    )
    corpus.valider(exiger_provenance=True)
    return corpus