"""Modeles metier de base, independants de l'API et des modeles ML."""

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class StatutSource(StrEnum):
    """Niveau de verification d'une source documentaire."""

    OFFICIEL = "officiel"
    INSTITUTIONNEL = "institutionnel"
    EXTERNE = "externe"


@dataclass(frozen=True)
class Source:
    titre: str
    origine: str
    url: str | None
    date_consultation: date
    statut: StatutSource
    donnees_extraites: str
    limites: str = ""
    incertitudes: str = ""
    identifiant: str = ""
    section: str = ""
    extrait: str = ""

    def __post_init__(self) -> None:
        if not self.titre.strip() or not self.origine.strip():
            raise ValueError("Le titre et l'origine de la source sont obligatoires.")
        if self.url is not None and not self.url.strip():
            raise ValueError("Une URL fournie doit être non vide.")


@dataclass(frozen=True)
class Matiere:
    identifiant: str
    nom: str
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Competence:
    identifiant: str
    nom: str
    description: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Prerequis:
    identifiant: str
    description: str
    obligatoire: bool = True
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Metier:
    identifiant: str
    nom: str
    description: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Parcours:
    identifiant: str
    nom: str
    matieres: tuple[str, ...] = field(default_factory=tuple)
    competences: tuple[str, ...] = field(default_factory=tuple)
    prerequis: tuple[str, ...] = field(default_factory=tuple)
    metiers: tuple[str, ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Mention:
    identifiant: str
    nom: str
    parcours: tuple[str, ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Formation:
    identifiant: str
    nom: str
    mentions: tuple[str, ...] = field(default_factory=tuple)
    niveau: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Passerelle:
    identifiant: str
    parcours_source: str
    parcours_cible: str
    description: str
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CorpusPedagogique:
    """Corpus versionné dont les relations sont contrôlées avant utilisation."""

    version: str
    mentions: tuple[Mention, ...] = field(default_factory=tuple)
    formations: tuple[Formation, ...] = field(default_factory=tuple)
    parcours: tuple[Parcours, ...] = field(default_factory=tuple)
    matieres: tuple[Matiere, ...] = field(default_factory=tuple)
    competences: tuple[Competence, ...] = field(default_factory=tuple)
    prerequis: tuple[Prerequis, ...] = field(default_factory=tuple)
    metiers: tuple[Metier, ...] = field(default_factory=tuple)
    passerelles: tuple[Passerelle, ...] = field(default_factory=tuple)
    sources: tuple[Source, ...] = field(default_factory=tuple)

    def valider(self, exiger_provenance: bool = False) -> None:
        """Lève une erreur si une relation pointe vers une entité absente."""
        collections = {
            "mention": self.mentions,
            "formation": self.formations,
            "parcours": self.parcours,
            "matiere": self.matieres,
            "competence": self.competences,
            "prerequis": self.prerequis,
            "metier": self.metiers,
            "passerelle": self.passerelles,
        }
        identifiants: dict[str, str] = {}
        for type_entite, entites in collections.items():
            for entite in entites:
                if entite.identifiant in identifiants:
                    raise ValueError(f"Identifiant dupliqué : {entite.identifiant}")
                identifiants[entite.identifiant] = type_entite

        sources = {source.identifiant for source in self.sources if source.identifiant}
        if len(sources) != len(self.sources):
            raise ValueError("Chaque source doit avoir un identifiant unique.")
        if exiger_provenance:
            for entites in collections.values():
                for entite in entites:
                    if not entite.sources:
                        raise ValueError(f"Provenance absente pour {entite.identifiant}")
                    if not set(entite.sources) <= sources:
                        raise ValueError(f"Source inconnue pour {entite.identifiant}")

        def verifier(reference: str, type_attendu: str, contexte: str) -> None:
            if identifiants.get(reference) != type_attendu:
                raise ValueError(f"Référence invalide dans {contexte} : {reference}")

        for mention in self.mentions:
            for parcours in mention.parcours:
                verifier(parcours, "parcours", mention.identifiant)
        for formation in self.formations:
            for mention in formation.mentions:
                verifier(mention, "mention", formation.identifiant)
        for parcours in self.parcours:
            for matiere in parcours.matieres:
                verifier(matiere, "matiere", parcours.identifiant)
            for competence in parcours.competences:
                verifier(competence, "competence", parcours.identifiant)
            for prerequis in parcours.prerequis:
                verifier(prerequis, "prerequis", parcours.identifiant)
            for metier in parcours.metiers:
                verifier(metier, "metier", parcours.identifiant)
        for passerelle in self.passerelles:
            verifier(passerelle.parcours_source, "parcours", passerelle.identifiant)
            verifier(passerelle.parcours_cible, "parcours", passerelle.identifiant)


@dataclass(frozen=True)
class Etudiant:
    identifiant: str
    competences: tuple[str, ...] = field(default_factory=tuple)
    matieres_preferees: tuple[str, ...] = field(default_factory=tuple)
    centres_interet: tuple[str, ...] = field(default_factory=tuple)
    niveau_etudes: str | None = None


@dataclass(frozen=True)
class ResultatModeleML:
    parcours: str
    score: float
    modele: str
    version: str

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("Le score ML doit être compris entre 0 et 1.")


@dataclass(frozen=True)
class ResultatRecommandation:
    """Sépare explicitement les décisions ML, documentaires et pédagogiques."""

    resultats_ml: tuple[ResultatModeleML, ...]
    informations_documentaires: tuple[str, ...] = field(default_factory=tuple)
    regles_pedagogiques: tuple[str, ...] = field(default_factory=tuple)
    explication_llm: str | None = None
    incertitude: str | None = None