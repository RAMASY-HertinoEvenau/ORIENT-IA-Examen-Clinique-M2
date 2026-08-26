"""Préparation reproductible des features du dataset V2, sans entraînement ML."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLONNE_CIBLE = "parcours_cible"
COLONNES_ENTREE = (
    "matieres_preferees",
    "moyenne_scolaire",
    "competences",
    "centres_interet",
    "projets",
    "preferences_professionnelles",
    "environnement_travail",
)
COLONNES_TEXTUELLES = ("matieres_preferees", "centres_interet", "projets")
COLONNES_CATEGORIELLES = ("preferences_professionnelles", "environnement_travail")
COLONNES_SYNTHETIQUES = (
    "matieres_preferees",
    "centres_interet",
    "projets",
    "preferences_professionnelles",
    "environnement_travail",
)


@dataclass(frozen=True)
class ConfigurationFeatures:
    """Décrit les groupes de variables inclus dans une matrice de features."""

    utiliser_moyenne: bool = True
    utiliser_competences: bool = True
    utiliser_variables_textuelles: bool = True
    utiliser_variables_synthetiques: bool = True


class TransformateurCompetences(BaseEstimator, TransformerMixin):
    """Convertit le JSON de compétences en colonnes numériques fixes."""

    def __init__(self, identifiants: tuple[str, ...]):
        self.identifiants = identifiants

    def fit(self, donnees: pd.DataFrame, cible: object = None) -> TransformateurCompetences:
        return self

    def transform(self, donnees: pd.DataFrame) -> np.ndarray:
        valeurs = donnees.iloc[:, 0].fillna("{}").map(json.loads)
        return np.asarray(
            [[float(dictionnaire.get(identifiant, 0)) for identifiant in self.identifiants]
             for dictionnaire in valeurs],
            dtype=float,
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray([f"competence_{identifiant}" for identifiant in self.identifiants])


class TransformateurListe(BaseEstimator, TransformerMixin):
    """Normalise une liste CSV séparée par des points-virgules pour le TF-IDF."""

    def __init__(self, valeur_vide: str = "aucun_projet_declare"):
        self.valeur_vide = valeur_vide

    def fit(self, donnees: object, cible: object = None) -> TransformateurListe:
        return self

    def transform(self, donnees: object) -> np.ndarray:
        valeurs = pd.Series(np.asarray(donnees).reshape(-1)).fillna("")
        return valeurs.map(
            lambda valeur: " ".join(
                element.strip() for element in str(valeur).split(";") if element.strip()
            ) or self.valeur_vide
        ).to_numpy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(["texte"])
        return np.asarray(input_features).reshape(-1)


@dataclass
class FeaturiseurML:
    """Construit et ajuste les transformations uniquement sur les données d'entraînement."""

    identifiants_competences: tuple[str, ...]
    configuration: ConfigurationFeatures = ConfigurationFeatures()

    def __post_init__(self) -> None:
        self._transformateur: ColumnTransformer | None = None
        self._colonnes_actives: tuple[str, ...] = ()

    def _construire_transformateur(self) -> ColumnTransformer:
        configuration = self.configuration
        transformations: list[tuple[str, Pipeline, list[str]]] = []
        colonnes_actives: list[str] = []

        if configuration.utiliser_moyenne:
            transformations.append((
                "moyenne",
                Pipeline([
                    ("imputation", SimpleImputer(strategy="median")),
                    ("standardisation", StandardScaler()),
                ]),
                ["moyenne_scolaire"],
            ))
            colonnes_actives.append("moyenne_scolaire")

        if configuration.utiliser_competences:
            transformations.append((
                "competences",
                Pipeline([
                    ("extraction", TransformateurCompetences(self.identifiants_competences)),
                    ("standardisation", StandardScaler()),
                ]),
                ["competences"],
            ))
            colonnes_actives.append("competences")

        if configuration.utiliser_variables_textuelles:
            for colonne in COLONNES_TEXTUELLES:
                if colonne == "projets" or configuration.utiliser_variables_synthetiques:
                    transformations.append((
                        f"texte_{colonne}",
                        Pipeline([
                            ("normalisation", TransformateurListe()),
                            ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),
                        ]),
                        [colonne],
                    ))
                    colonnes_actives.append(colonne)

        if configuration.utiliser_variables_synthetiques:
            transformations.append((
                "categories",
                Pipeline([
                    ("imputation", SimpleImputer(strategy="most_frequent")),
                    ("encodage", OneHotEncoder(handle_unknown="ignore")),
                ]),
                list(COLONNES_CATEGORIELLES),
            ))
            colonnes_actives.extend(COLONNES_CATEGORIELLES)

        if not transformations:
            raise ValueError("La configuration ne contient aucune feature.")
        self._colonnes_actives = tuple(colonnes_actives)
        return ColumnTransformer(transformations, remainder="drop")

    def fit(self, donnees: pd.DataFrame, cible: object = None) -> FeaturiseurML:
        self._verifier_donnees(donnees)
        self._transformateur = self._construire_transformateur()
        self._transformateur.fit(donnees)
        return self

    def transform(self, donnees: pd.DataFrame):
        self._verifier_donnees(donnees)
        if self._transformateur is None:
            raise RuntimeError("Le featuriseur doit être ajusté sur le train avant transform().")
        return self._transformateur.transform(donnees)

    def fit_transform(self, donnees: pd.DataFrame, cible: object = None):
        self.fit(donnees, cible)
        return self.transform(donnees)

    def get_feature_names_out(self) -> np.ndarray:
        if self._transformateur is None:
            raise RuntimeError("Le featuriseur doit être ajusté avant get_feature_names_out().")
        return self._transformateur.get_feature_names_out()

    def _verifier_donnees(self, donnees: pd.DataFrame) -> None:
        colonnes_interdites = {COLONNE_CIBLE, "id_candidat"}
        colonnes_absentes = set(COLONNES_ENTREE) - set(donnees.columns)
        if colonnes_absentes:
            raise ValueError(f"Colonnes d'entrée absentes: {sorted(colonnes_absentes)}")
        if set(donnees.columns) & colonnes_interdites - {COLONNE_CIBLE}:
            raise ValueError("id_candidat ne peut pas être utilisé comme feature.")


def charger_splits(dossier: Path | str = "data/full_sample_2000_v2") -> dict[str, pd.DataFrame]:
    """Charge les trois splits V2 sans les modifier."""
    dossier = Path(dossier)
    return {
        nom_split: pd.read_csv(dossier / f"{nom_split}.csv")
        for nom_split in ("train", "val", "test")
    }


def separer_cible(donnees: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare strictement la cible des entrées et exclut l'identifiant candidat."""
    if COLONNE_CIBLE not in donnees or "id_candidat" not in donnees:
        raise ValueError("Les colonnes parcours_cible et id_candidat sont obligatoires.")
    entrees = donnees.drop(columns=[COLONNE_CIBLE, "id_candidat"]).copy()
    cible = donnees[COLONNE_CIBLE].copy()
    return entrees, cible


def configurations_ablations() -> dict[str, ConfigurationFeatures]:
    """Retourne les cinq configurations prévues, sans ajuster de transformation."""
    return {
        "toutes_variables": ConfigurationFeatures(),
        "sans_moyenne": ConfigurationFeatures(utiliser_moyenne=False),
        "sans_competences": ConfigurationFeatures(utiliser_competences=False),
        "sans_variables_textuelles": ConfigurationFeatures(utiliser_variables_textuelles=False),
        "variables_synthetiques_uniquement": ConfigurationFeatures(
            utiliser_moyenne=False,
            utiliser_competences=False,
        ),
    }


def preparer_splits(
    dossier: Path | str = "data/full_sample_2000_v2",
    configuration: ConfigurationFeatures | None = None,
    identifiants_competences: tuple[str, ...] = (),
) -> tuple[dict[str, object], dict[str, pd.Series], FeaturiseurML]:
    """Ajuste sur train puis transforme train, validation et test en mémoire."""
    splits = charger_splits(dossier)
    entrees_train, cible_train = separer_cible(splits["train"])
    identifiants = identifiants_competences or tuple(
        json.loads(splits["train"].iloc[0]["competences"]).keys()
    )
    featuriseur = FeaturiseurML(identifiants, configuration or ConfigurationFeatures())
    matrices = {"train": featuriseur.fit_transform(entrees_train)}
    cibles = {"train": cible_train}
    for nom_split in ("val", "test"):
        entrees, cible = separer_cible(splits[nom_split])
        matrices[nom_split] = featuriseur.transform(entrees)
        cibles[nom_split] = cible
    return matrices, cibles, featuriseur
