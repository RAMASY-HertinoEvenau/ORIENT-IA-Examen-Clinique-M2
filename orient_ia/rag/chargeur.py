"""Chargement et préparation du corpus pédagogique pour le RAG."""
from __future__ import annotations

from pathlib import Path

from orient_ia.domaine.corpus import charger_corpus
from orient_ia.domaine.modeles import (
    CorpusPedagogique,
    Formation,
    Metier,
    Parcours,
    Prerequis,
    Source,
)
from orient_ia.rag.modeles import DocumentRAG

_CHEMIN_CORPUS = Path(__file__).resolve().parents[2] / "donnees" / "corpus_pedagogique.json"


def _charger_sources(corpus: CorpusPedagogique) -> dict[str, Source]:
    return {source.identifiant: source for source in corpus.sources}


def _document_formation(formation: Formation, sources: dict[str, Source]) -> DocumentRAG:
    source = sources.get(formation.sources[0]) if formation.sources else None
    contenu = (
        f"Formation {formation.nom} ({formation.identifiant}). "
        f"Niveau: {formation.niveau or 'non communiqué'}. "
        f"Mentions associées: {', '.join(formation.mentions) if formation.mentions else 'non communiqué'}."
    )
    return DocumentRAG(
        identifiant=formation.identifiant,
        titre=formation.nom,
        type_document="formation",
        contenu=contenu,
        source=source.titre if source else "source introuvable",
        url=source.url if source else None,
        date_consultation=source.date_consultation.isoformat() if source else None,
        statut=source.statut.value if source else None,
        identifiant_source=source.identifiant if source else "",
        metadonnees={"niveau": formation.niveau, "mentions": list(formation.mentions)},
    )


def _document_parcours(parcours: Parcours, sources: dict[str, Source]) -> DocumentRAG:
    source = sources.get(parcours.sources[0]) if parcours.sources else None
    prerequis = " ; ".join(parcours.prerequis) if parcours.prerequis else "Aucun prérequis explicitement déclaré."
    competences = " ; ".join(parcours.competences) if parcours.competences else "Aucune compétence explicitement déclarée."
    metiers = " ; ".join(parcours.metiers) if parcours.metiers else "Aucun métier explicitement déclaré."
    contenu = (
        f"Parcours {parcours.nom} ({parcours.identifiant}). "
        f"Prérequis: {prerequis}. "
        f"Compétences: {competences}. "
        f"Métiers associés: {metiers}."
    )
    return DocumentRAG(
        identifiant=parcours.identifiant,
        titre=parcours.nom,
        type_document="parcours",
        contenu=contenu,
        source=source.titre if source else "source introuvable",
        url=source.url if source else None,
        date_consultation=source.date_consultation.isoformat() if source else None,
        statut=source.statut.value if source else None,
        identifiant_source=source.identifiant if source else "",
        metadonnees={
            "matieres": list(parcours.matieres),
            "competences": list(parcours.competences),
            "prerequis": list(parcours.prerequis),
            "metiers": list(parcours.metiers),
        },
    )


def _document_prerequis(prerequis: Prerequis, sources: dict[str, Source]) -> DocumentRAG:
    source = sources.get(prerequis.sources[0]) if prerequis.sources else None
    contenu = (
        f"Prérequis {prerequis.identifiant}. {prerequis.description} "
        f"Obligatoire: {'oui' if prerequis.obligatoire else 'non'}."
    )
    return DocumentRAG(
        identifiant=prerequis.identifiant,
        titre=prerequis.identifiant,
        type_document="prerequis",
        contenu=contenu,
        source=source.titre if source else "source introuvable",
        url=source.url if source else None,
        date_consultation=source.date_consultation.isoformat() if source else None,
        statut=source.statut.value if source else None,
        identifiant_source=source.identifiant if source else "",
        metadonnees={"obligatoire": prerequis.obligatoire},
    )


def _document_metier(metier: Metier, sources: dict[str, Source]) -> DocumentRAG:
    source = sources.get(metier.sources[0]) if metier.sources else None
    contenu = f"Métier {metier.nom} ({metier.identifiant}). {metier.description}"
    return DocumentRAG(
        identifiant=metier.identifiant,
        titre=metier.nom,
        type_document="metier",
        contenu=contenu,
        source=source.titre if source else "source introuvable",
        url=source.url if source else None,
        date_consultation=source.date_consultation.isoformat() if source else None,
        statut=source.statut.value if source else None,
        identifiant_source=source.identifiant if source else "",
        metadonnees={"description": metier.description},
    )


def _document_source(source: Source) -> DocumentRAG:
    contenu = (
        f"Source {source.titre} ({source.identifiant}). "
        f"Origine: {source.origine}. "
        f"Section: {source.section or 'non précisée'}. "
        f"Extrait: {source.extrait or source.donnees_extraites}."
    )
    return DocumentRAG(
        identifiant=source.identifiant,
        titre=source.titre,
        type_document="source",
        contenu=contenu,
        source=source.titre,
        url=source.url,
        date_consultation=source.date_consultation.isoformat(),
        statut=source.statut.value,
        identifiant_source=source.identifiant,
        metadonnees={
            "origine": source.origine,
            "section": source.section,
            "donnees_extraites": source.donnees_extraites,
            "limites": source.limites,
            "incertitudes": source.incertitudes,
        },
    )


def charger_documents_pedagogiques(chemin: str | Path | None = None) -> list[DocumentRAG]:
    """Charge le corpus pédagogique et retourne des documents exploitables par le RAG."""
    chemin_corpus = Path(chemin) if chemin is not None else _CHEMIN_CORPUS
    corpus = charger_corpus(chemin_corpus)
    sources = _charger_sources(corpus)
    documents: list[DocumentRAG] = []

    for source in corpus.sources:
        documents.append(_document_source(source))
    for formation in corpus.formations:
        documents.append(_document_formation(formation, sources))
    for parcours in corpus.parcours:
        documents.append(_document_parcours(parcours, sources))
    for prerequis in corpus.prerequis:
        documents.append(_document_prerequis(prerequis, sources))
    for metier in corpus.metiers:
        documents.append(_document_metier(metier, sources))

    return sorted(documents, key=lambda document: document.identifiant)
