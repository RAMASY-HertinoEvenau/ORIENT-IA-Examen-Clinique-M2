"""Validation indépendante sur val.csv après sélection par validation croisée."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from orient_ia.ml.experimentations import _construire_modeles, _identifiants_competences
from orient_ia.ml.featurisation import (
    COLONNE_CIBLE,
    ConfigurationFeatures,
    FeaturiseurML,
    configurations_ablations,
    separer_cible,
)

SEED_VALIDATION_INDEPENDANTE = 42


def configurations_candidats_finaux() -> dict[str, tuple[str, ConfigurationFeatures]]:
    """Retourne exactement les trois candidats conservés après la validation croisée."""
    ablations = configurations_ablations()
    return {
        "extra_trees_sans_competences": ("extra_trees", ablations["sans_competences"]),
        "svm_lineaire_sans_competences": ("svm_lineaire", ablations["sans_competences"]),
        "extra_trees_sans_moyenne": ("extra_trees", ablations["sans_moyenne"]),
    }


def _evaluer(
    cible_reelle: pd.Series,
    predictions: np.ndarray,
    classes: np.ndarray,
    probabilites: np.ndarray | None,
) -> dict[str, Any]:
    precision, rappel, _, _ = precision_recall_fscore_support(
        cible_reelle,
        predictions,
        labels=classes,
        zero_division=0,
    )
    resultat = {
        "f1_macro": round(float(f1_score(cible_reelle, predictions, average="macro", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(cible_reelle, predictions)), 6),
        "accuracy": round(float(accuracy_score(cible_reelle, predictions)), 6),
        "precision_macro": round(float(np.mean(precision)), 6),
        "rappel_macro": round(float(np.mean(rappel)), 6),
        "precision_par_classe": dict(zip(classes.tolist(), precision.round(6).tolist())),
        "rappel_par_classe": dict(zip(classes.tolist(), rappel.round(6).tolist())),
        "nombre_classes_rappel_nul": int(np.sum(rappel == 0)),
        "classes": classes.tolist(),
        "matrice_confusion": confusion_matrix(cible_reelle, predictions, labels=classes).tolist(),
    }
    if probabilites is not None:
        resultat["log_loss"] = round(float(log_loss(cible_reelle, probabilites, labels=classes)), 6)
    return resultat


def executer_validation_independante(
    dossier: Path | str = "data/full_sample_2000_v2",
    seed: int = SEED_VALIDATION_INDEPENDANTE,
) -> dict[str, Any]:
    """Ajuste sur train.csv et évalue les trois candidats sur val.csv uniquement."""
    dossier = Path(dossier)
    train = pd.read_csv(dossier / "train.csv")
    validation = pd.read_csv(dossier / "val.csv")
    classes = np.asarray(sorted(train[COLONNE_CIBLE].unique()))
    identifiants = _identifiants_competences(train)
    resultats = {}

    for nom, (nom_modele, configuration) in configurations_candidats_finaux().items():
        entrees_train, cible_train = separer_cible(train)
        entrees_validation, cible_validation = separer_cible(validation)
        featuriseur = FeaturiseurML(identifiants, configuration)
        matrice_train = featuriseur.fit_transform(entrees_train)
        matrice_validation = featuriseur.transform(entrees_validation)
        modele = _construire_modeles(seed)[nom_modele]
        modele.fit(matrice_train, cible_train)
        predictions = modele.predict(matrice_validation)
        probabilites = modele.predict_proba(matrice_validation) if hasattr(modele, "predict_proba") else None
        resultats[nom] = _evaluer(cible_validation, predictions, classes, probabilites)
        resultats[nom]["nombre_features"] = int(matrice_train.shape[1])

    return {
        "dataset": "data/full_sample_2000_v2",
        "fichiers_utilises": ["train.csv", "val.csv"],
        "fichier_interdit": "test.csv",
        "seed": seed,
        "nombre_classes": len(classes),
        "candidats": resultats,
    }


if __name__ == "__main__":
    print(json.dumps(executer_validation_independante(), ensure_ascii=False, indent=2))
