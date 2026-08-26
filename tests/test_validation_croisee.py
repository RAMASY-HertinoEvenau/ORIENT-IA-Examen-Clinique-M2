import json
from pathlib import Path

from orient_ia.ml import validation_croisee
from orient_ia.ml.validation_croisee import executer_validation_croisee

CHEMIN_DATASET = Path("data/full_sample_2000_v2")


def test_validation_croisee_utilise_cinq_folds_et_trois_seeds() -> None:
    resultat = executer_validation_croisee(CHEMIN_DATASET, seeds=(11, 22, 33), nombre_folds=5)

    assert resultat["fichier_utilise"] == "train.csv"
    assert resultat["fichier_interdit"] == "test.csv"
    assert resultat["seeds"] == [11, 22, 33]
    assert resultat["nombre_folds"] == 5
    assert resultat["nombre_classes"] == 16
    assert all(
        experience["nombre_folds"] == 15
        for experience in resultat["experiences"].values()
    )


def test_validation_croisee_expose_moyenne_ecart_type_intervalle_et_folds() -> None:
    resultat = executer_validation_croisee(CHEMIN_DATASET, seeds=(11,), nombre_folds=2)
    resume = resultat["experiences"]["extra_trees_sans_competences"]

    for metrique in ("f1_macro", "balanced_accuracy", "accuracy"):
        assert set(resume[metrique]) == {"moyenne", "ecart_type", "intervalle_confiance_95"}
    assert len(resume["resultats_par_fold_seed"]) == 2
    assert set(resume["rappel_par_classe"]) == set(resume["classes"])
    assert "nombre_classes_rappel_nul" in resume


def test_validation_croisee_ne_charge_pas_test(monkeypatch) -> None:
    lecture_originale = validation_croisee.pd.read_csv
    chemins_lus = []

    def lire_csv(chemin, *args, **kwargs):
        chemins_lus.append(str(chemin))
        assert not str(chemin).endswith("test.csv")
        return lecture_originale(chemin, *args, **kwargs)

    monkeypatch.setattr(validation_croisee.pd, "read_csv", lire_csv)
    executer_validation_croisee(CHEMIN_DATASET, seeds=(11,), nombre_folds=2)

    assert all(not chemin.endswith("test.csv") for chemin in chemins_lus)
    assert chemins_lus == [str(CHEMIN_DATASET / "train.csv")]


def test_validation_croisee_est_reproductible() -> None:
    premier = executer_validation_croisee(CHEMIN_DATASET, seeds=(11,), nombre_folds=2)
    second = executer_validation_croisee(CHEMIN_DATASET, seeds=(11,), nombre_folds=2)

    assert json.dumps(premier, sort_keys=True) == json.dumps(second, sort_keys=True)
