from pathlib import Path

import numpy as np

from orient_ia.ml import experimentations
from orient_ia.ml.experimentations import (
    ClassifieurCentroides,
    _predire_aleatoire,
    _predire_majoritaire,
    executer_experiences,
)
from orient_ia.ml.featurisation import charger_splits, separer_cible

CHEMIN_DATASET = Path("data/full_sample_2000_v2")


def test_baselines_produisent_des_predictions_valides() -> None:
    train = charger_splits(CHEMIN_DATASET)["train"]
    validation = charger_splits(CHEMIN_DATASET)["val"]
    classes = np.asarray(sorted(train["parcours_cible"].unique()))

    prediction_majoritaire, probabilite_majoritaire = _predire_majoritaire(
        train["parcours_cible"], len(validation), classes
    )
    prediction_aleatoire, probabilite_aleatoire = _predire_aleatoire(
        train["parcours_cible"], len(validation), classes, seed=42
    )

    assert len(prediction_majoritaire) == len(validation)
    assert len(prediction_aleatoire) == len(validation)
    assert probabilite_majoritaire.shape == probabilite_aleatoire.shape
    assert set(prediction_aleatoire).issubset(set(classes))


def test_evaluation_contient_les_seize_classes_et_aucune_feature_cible() -> None:
    resultats = executer_experiences(CHEMIN_DATASET)

    assert len(resultats) == 20
    assert all(len(resultat["classes"]) == 16 for resultat in resultats)
    assert all("parcours_cible" not in nom for resultat in resultats for nom in resultat)


def test_experimentation_ne_lit_jamais_le_test(monkeypatch) -> None:
    lecture_originale = experimentations.pd.read_csv
    chemins_lus = []

    def lire_csv(chemin, *args, **kwargs):
        chemins_lus.append(str(chemin))
        assert not str(chemin).endswith("test.csv")
        return lecture_originale(chemin, *args, **kwargs)

    monkeypatch.setattr(experimentations.pd, "read_csv", lire_csv)
    executer_experiences(CHEMIN_DATASET)

    assert all(not chemin.endswith("test.csv") for chemin in chemins_lus)
    assert any(chemin.endswith("train.csv") for chemin in chemins_lus)
    assert any(chemin.endswith("val.csv") for chemin in chemins_lus)


def test_experimentation_est_reproductible() -> None:
    premier_resultat = executer_experiences(CHEMIN_DATASET, seed=42)
    second_resultat = executer_experiences(CHEMIN_DATASET, seed=42)

    assert premier_resultat == second_resultat


def test_centroides_fonctionnent_sur_une_matrice_transformee() -> None:
    train = charger_splits(CHEMIN_DATASET)["train"].head(100)
    entrees, cible = separer_cible(train)
    from orient_ia.ml.featurisation import FeaturiseurML

    identifiants = tuple(__import__("json").loads(train.iloc[0]["competences"]).keys())
    featuriseur = FeaturiseurML(identifiants)
    matrice = featuriseur.fit_transform(entrees)
    classifieur = ClassifieurCentroides().fit(matrice, cible)

    assert len(classifieur.predict(matrice)) == len(train)
