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

    def __post_init__(self) -> None:
        if not self.titre.strip() or not self.origine.strip():
            raise ValueError("Le titre et l'origine de la source sont obligatoires.")
        if self.url is not None and not self.url.strip():
            raise ValueError("Une URL fournie doit être non vide.")


@dataclass(frozen=True)
class Matiere:
    identifiant: str
    nom: str


@dataclass(frozen=True)
class Competence:
    identifiant: str
    nom: str
    description: str = ""


@dataclass(frozen=True)
class Prerequis:
    identifiant: str
    description: str
    obligatoire: bool = True


@dataclass(frozen=True)
class Metier:
    identifiant: str
    nom: str
    description: str = ""


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


@dataclass(frozen=True)
class Formation:
    identifiant: str
    nom: str
    mentions: tuple[str, ...] = field(default_factory=tuple)
    niveau: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)


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