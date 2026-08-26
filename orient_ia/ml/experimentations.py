"""Baselines et expérimentations ML limitées au train et à la validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from orient_ia.ml.featurisation import (
    COLONNE_CIBLE,
    ConfigurationFeatures,
    FeaturiseurML,
    configurations_ablations,
    separer_cible,
)

SEED_EXPERIENCE = 42


@dataclass(frozen=True)
class Experience:
    """Décrit une expérience et sa configuration de variables."""

    nom: str
    configuration: ConfigurationFeatures


def configurations_baselines() -> dict[str, ConfigurationFeatures]:
    """Retourne les configurations minimales des baselines structurées."""
    return {
        "moyenne_seule": ConfigurationFeatures(
            utiliser_competences=False,
            utiliser_variables_textuelles=False,
            utiliser_variables_synthetiques=False,
        ),
        "competences_seules": ConfigurationFeatures(
            utiliser_moyenne=False,
            utiliser_variables_textuelles=False,
            utiliser_variables_synthetiques=False,
        ),
        "variables_synthetiques_uniquement": configurations_ablations()[
            "variables_synthetiques_uniquement"
        ],
    }


def _matrice_dense(matrice: Any) -> np.ndarray:
    return matrice.toarray() if hasattr(matrice, "toarray") else np.asarray(matrice)


def _evaluer_predictions(
    cible_reelle: pd.Series,
    predictions: np.ndarray,
    classes: np.ndarray,
    probabilites: np.ndarray | None = None,
) -> dict[str, Any]:
    precision, rappel, _, _ = precision_recall_fscore_support(
        cible_reelle,
        predictions,
        labels=classes,
        zero_division=0,
    )
    resultat: dict[str, Any] = {
        "f1_macro": round(f1_score(cible_reelle, predictions, average="macro", zero_division=0), 6),
        "balanced_accuracy": round(balanced_accuracy_score(cible_reelle, predictions), 6),
        "accuracy": round(accuracy_score(cible_reelle, predictions), 6),
        "precision_par_classe": dict(zip(classes.tolist(), precision.round(6).tolist())),
        "rappel_par_classe": dict(zip(classes.tolist(), rappel.round(6).tolist())),
        "classes": classes.tolist(),
        "matrice_confusion": confusion_matrix(cible_reelle, predictions, labels=classes).tolist(),
    }
    if probabilites is not None:
        resultat["log_loss"] = round(log_loss(cible_reelle, probabilites, labels=classes), 6)
    return resultat


def _entrainer_et_evaluer(
    nom: str,
    estimateur: Any,
    matrice_train: Any,
    cible_train: pd.Series,
    matrice_validation: Any,
    cible_validation: pd.Series,
    classes: np.ndarray,
) -> dict[str, Any]:
    estimateur.fit(matrice_train, cible_train)
    predictions = estimateur.predict(matrice_validation)
    probabilites = estimateur.predict_proba(matrice_validation) if hasattr(estimateur, "predict_proba") else None
    resultat = _evaluer_predictions(cible_validation, predictions, classes, probabilites)
    resultat.update({"experience": nom, "type": "modele"})
    return resultat


def _predire_majoritaire(cible_train: pd.Series, taille: int, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    majoritaire = cible_train.value_counts().index[0]
    predictions = np.full(taille, majoritaire, dtype=object)
    probabilites = np.zeros((taille, len(classes)))
    probabilites[:, np.where(classes == majoritaire)[0][0]] = 1.0
    return predictions, probabilites


def _predire_aleatoire(
    cible_train: pd.Series,
    taille: int,
    classes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    proportions = cible_train.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    generateur = np.random.default_rng(seed)
    predictions = generateur.choice(classes, size=taille, p=proportions)
    probabilites = np.tile(proportions, (taille, 1))
    return predictions, probabilites


class ClassifieurCentroides:
    """Baseline sans apprentissage complexe : centroïde numérique par classe."""

    def fit(self, matrice: Any, cible: pd.Series) -> ClassifieurCentroides:
        valeurs = _matrice_dense(matrice)
        self.classes_ = np.asarray(sorted(cible.unique()))
        self.centroides_ = np.asarray(
            [valeurs[cible.to_numpy() == classe].mean(axis=0) for classe in self.classes_]
        )
        return self

    def predict(self, matrice: Any) -> np.ndarray:
        valeurs = _matrice_dense(matrice)
        distances = ((valeurs[:, None, :] - self.centroides_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[distances.argmin(axis=1)]


def _construire_modeles(seed: int) -> dict[str, Any]:
    return {
        "regression_logistique": LogisticRegression(max_iter=1000, random_state=seed),
        "svm_lineaire": LinearSVC(random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    }


def _identifiants_competences(train: pd.DataFrame) -> tuple[str, ...]:
    return tuple(json.loads(train.iloc[0]["competences"]).keys())


def executer_experiences(
    dossier: Path | str = "data/full_sample_2000_v2",
    seed: int = SEED_EXPERIENCE,
) -> list[dict[str, Any]]:
    """Compare baselines et modèles sur train/validation uniquement."""
    dossier = Path(dossier)
    train = pd.read_csv(dossier / "train.csv")
    validation = pd.read_csv(dossier / "val.csv")
    classes = np.asarray(sorted(train[COLONNE_CIBLE].unique()))
    identifiants = _identifiants_competences(train)
    resultats: list[dict[str, Any]] = []

    predictions, probabilites = _predire_majoritaire(train[COLONNE_CIBLE], len(validation), classes)
    resultat = _evaluer_predictions(validation[COLONNE_CIBLE], predictions, classes, probabilites)
    resultat.update({"experience": "classe_majoritaire", "type": "baseline"})
    resultats.append(resultat)

    predictions, probabilites = _predire_aleatoire(
        train[COLONNE_CIBLE], len(validation), classes, seed
    )
    resultat = _evaluer_predictions(validation[COLONNE_CIBLE], predictions, classes, probabilites)
    resultat.update({"experience": "aleatoire_stratifiee", "type": "baseline"})
    resultats.append(resultat)

    for nom, configuration in configurations_baselines().items():
        entrees_train, cible_train = separer_cible(train)
        entrees_validation, cible_validation = separer_cible(validation)
        featuriseur = FeaturiseurML(identifiants, configuration)
        matrice_train = featuriseur.fit_transform(entrees_train)
        matrice_validation = featuriseur.transform(entrees_validation)
        resultat = _entrainer_et_evaluer(
            f"baseline_centroides_{nom}",
            ClassifieurCentroides(),
            matrice_train,
            cible_train,
            matrice_validation,
            cible_validation,
            classes,
        )
        resultat["type"] = "baseline"
        resultats.append(resultat)

    for nom_configuration, configuration in configurations_ablations().items():
        entrees_train, cible_train = separer_cible(train)
        entrees_validation, cible_validation = separer_cible(validation)
        featuriseur = FeaturiseurML(identifiants, configuration)
        matrice_train = featuriseur.fit_transform(entrees_train)
        matrice_validation = featuriseur.transform(entrees_validation)
        for nom_modele, modele in _construire_modeles(seed).items():
            resultat = _entrainer_et_evaluer(
                f"{nom_modele}_{nom_configuration}",
                modele,
                matrice_train,
                cible_train,
                matrice_validation,
                cible_validation,
                classes,
            )
            resultat["configuration"] = nom_configuration
            resultats.append(resultat)
    return resultats
