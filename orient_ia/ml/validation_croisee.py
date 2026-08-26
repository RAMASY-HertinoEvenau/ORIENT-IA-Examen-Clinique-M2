"""Validation croisée stratifiée des candidats ML sur train.csv uniquement."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from orient_ia.ml.experimentations import (
    ClassifieurCentroides,
    _construire_modeles,
    _evaluer_predictions,
    _identifiants_competences,
    _predire_aleatoire,
    _predire_majoritaire,
)
from orient_ia.ml.featurisation import (
    COLONNE_CIBLE,
    ConfigurationFeatures,
    FeaturiseurML,
    configurations_ablations,
    separer_cible,
)

SEEDS_VALIDATION_CROISEE = (42, 43, 44)
NOMBRE_FOLDS = 5


def configurations_candidats() -> dict[str, tuple[str, ConfigurationFeatures]]:
    """Définit les candidats à conserver avant la validation indépendante."""
    ablations = configurations_ablations()
    return {
        "extra_trees_sans_competences": ("extra_trees", ablations["sans_competences"]),
        "extra_trees_sans_moyenne": ("extra_trees", ablations["sans_moyenne"]),
        "extra_trees_variables_synthetiques": (
            "extra_trees",
            ablations["variables_synthetiques_uniquement"],
        ),
        "svm_lineaire_sans_competences": ("svm_lineaire", ablations["sans_competences"]),
    }


def configurations_baselines_croisees() -> dict[str, ConfigurationFeatures]:
    """Définit les baselines comparées dans les mêmes folds."""
    ablations = configurations_ablations()
    return {
        "baseline_centroides_moyenne": ConfigurationFeatures(
            utiliser_competences=False,
            utiliser_variables_textuelles=False,
            utiliser_variables_synthetiques=False,
        ),
        "baseline_centroides_competences": ConfigurationFeatures(
            utiliser_moyenne=False,
            utiliser_variables_textuelles=False,
            utiliser_variables_synthetiques=False,
        ),
        "baseline_centroides_synthetiques": ablations["variables_synthetiques_uniquement"],
    }


def _intervalle_confiance(valeurs: list[float]) -> tuple[float, float]:
    moyenne = float(np.mean(valeurs))
    erreur = 1.96 * float(np.std(valeurs, ddof=1)) / np.sqrt(len(valeurs)) if len(valeurs) > 1 else 0.0
    return round(moyenne - erreur, 6), round(moyenne + erreur, 6)


def _resume_metrique(resultats: list[dict[str, Any]], cle: str) -> dict[str, Any]:
    valeurs = [float(resultat[cle]) for resultat in resultats]
    return {
        "moyenne": round(float(np.mean(valeurs)), 6),
        "ecart_type": round(float(np.std(valeurs, ddof=1)), 6) if len(valeurs) > 1 else 0.0,
        "intervalle_confiance_95": _intervalle_confiance(valeurs),
    }


def _creer_estimateur(nom_modele: str, seed: int) -> Any:
    if nom_modele == "centroides":
        return ClassifieurCentroides()
    return _construire_modeles(seed)[nom_modele]


def _evaluer_fold(
    nom_experience: str,
    nom_modele: str,
    configuration: ConfigurationFeatures,
    train: pd.DataFrame,
    indices_apprentissage: np.ndarray,
    indices_validation: np.ndarray,
    identifiants_competences: tuple[str, ...],
    classes: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    donnees_apprentissage = train.iloc[indices_apprentissage]
    donnees_validation = train.iloc[indices_validation]
    entrees_apprentissage, cible_apprentissage = separer_cible(donnees_apprentissage)
    entrees_validation, cible_validation = separer_cible(donnees_validation)
    featuriseur = FeaturiseurML(identifiants_competences, configuration)
    matrice_apprentissage = featuriseur.fit_transform(entrees_apprentissage)
    matrice_validation = featuriseur.transform(entrees_validation)
    estimateur = _creer_estimateur(nom_modele, seed)
    estimateur.fit(matrice_apprentissage, cible_apprentissage)
    predictions = estimateur.predict(matrice_validation)
    resultats = _evaluer_predictions(cible_validation, predictions, classes)
    resultats.update({"seed": seed, "fold": int(seed), "experience": nom_experience})
    return resultats


def _evaluer_baseline_majoritaire(
    train: pd.DataFrame,
    indices_apprentissage: np.ndarray,
    indices_validation: np.ndarray,
    classes: np.ndarray,
    seed: int,
    nom: str,
) -> dict[str, Any]:
    cible_apprentissage = train.iloc[indices_apprentissage][COLONNE_CIBLE]
    cible_validation = train.iloc[indices_validation][COLONNE_CIBLE]
    predictions, probabilites = _predire_majoritaire(cible_apprentissage, len(cible_validation), classes)
    resultat = _evaluer_predictions(cible_validation, predictions, classes, probabilites)
    resultat.update({"seed": seed, "fold": int(seed), "experience": nom})
    return resultat


def _evaluer_baseline_aleatoire(
    train: pd.DataFrame,
    indices_apprentissage: np.ndarray,
    indices_validation: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    cible_apprentissage = train.iloc[indices_apprentissage][COLONNE_CIBLE]
    cible_validation = train.iloc[indices_validation][COLONNE_CIBLE]
    predictions, probabilites = _predire_aleatoire(
        cible_apprentissage, len(cible_validation), classes, seed
    )
    resultat = _evaluer_predictions(cible_validation, predictions, classes, probabilites)
    resultat.update({"seed": seed, "fold": int(seed), "experience": "baseline_aleatoire_stratifiee"})
    return resultat


def executer_validation_croisee(
    dossier: Path | str = "data/full_sample_2000_v2",
    seeds: tuple[int, ...] = SEEDS_VALIDATION_CROISEE,
    nombre_folds: int = NOMBRE_FOLDS,
) -> dict[str, Any]:
    """Exécute les folds sur train.csv, sans charger val.csv ni test.csv."""
    train = pd.read_csv(Path(dossier) / "train.csv")
    classes = np.asarray(sorted(train[COLONNE_CIBLE].unique()))
    identifiants = _identifiants_competences(train)
    resultats_par_experience: dict[str, list[dict[str, Any]]] = {}
    toutes_les_experiences = {
        "baseline_classe_majoritaire": ("majoritaire", ConfigurationFeatures()),
        "baseline_aleatoire_stratifiee": ("aleatoire", ConfigurationFeatures()),
    }
    toutes_les_experiences.update(
        {nom: ("centroides", configuration) for nom, configuration in configurations_baselines_croisees().items()}
    )
    toutes_les_experiences.update(configurations_candidats())

    for seed in seeds:
        decoupage = StratifiedKFold(n_splits=nombre_folds, shuffle=True, random_state=seed)
        for numero_fold, (indices_apprentissage, indices_validation) in enumerate(
            decoupage.split(train, train[COLONNE_CIBLE]), start=1
        ):
            for nom, (nom_modele, configuration) in toutes_les_experiences.items():
                if nom_modele == "majoritaire":
                    resultat = _evaluer_baseline_majoritaire(
                        train, indices_apprentissage, indices_validation, classes, seed, nom
                    )
                elif nom_modele == "aleatoire":
                    resultat = _evaluer_baseline_aleatoire(
                        train, indices_apprentissage, indices_validation, classes, seed + numero_fold
                    )
                else:
                    resultat = _evaluer_fold(
                        nom,
                        nom_modele,
                        configuration,
                        train,
                        indices_apprentissage,
                        indices_validation,
                        identifiants,
                        classes,
                        seed,
                    )
                resultat["fold"] = numero_fold
                resultats_par_experience.setdefault(nom, []).append(resultat)

    resumes = {}
    for nom, resultats in resultats_par_experience.items():
        rappels = {
            classe: {
                "moyenne": round(float(np.mean([r["rappel_par_classe"][classe] for r in resultats])), 6),
                "ecart_type": round(float(np.std([r["rappel_par_classe"][classe] for r in resultats], ddof=1)), 6),
            }
            for classe in classes
        }
        resumes[nom] = {
            "nombre_folds": len(resultats),
            "classes": classes.tolist(),
            "f1_macro": _resume_metrique(resultats, "f1_macro"),
            "balanced_accuracy": _resume_metrique(resultats, "balanced_accuracy"),
            "accuracy": _resume_metrique(resultats, "accuracy"),
            "rappel_par_classe": rappels,
            "nombre_classes_rappel_nul": {
                "moyenne": round(float(np.mean([sum(v == 0 for v in r["rappel_par_classe"].values()) for r in resultats])), 6),
                "maximum": max(sum(v == 0 for v in r["rappel_par_classe"].values()) for r in resultats),
            },
            "resultats_par_fold_seed": resultats,
        }
    return {
        "dataset": "data/full_sample_2000_v2",
        "fichier_utilise": "train.csv",
        "fichier_interdit": "test.csv",
        "seeds": list(seeds),
        "nombre_folds": nombre_folds,
        "nombre_classes": len(classes),
        "experiences": resumes,
    }
