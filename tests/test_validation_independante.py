from pathlib import Path

from orient_ia.ml import validation_independante
from orient_ia.ml.validation_independante import executer_validation_independante

CHEMIN_DATASET = Path("data/full_sample_2000_v2")


def test_validation_independante_utilise_train_et_val_uniquement() -> None:
    resultat = executer_validation_independante(CHEMIN_DATASET)

    assert resultat["fichiers_utilises"] == ["train.csv", "val.csv"]
    assert resultat["fichier_interdit"] == "test.csv"
    assert resultat["seed"] == 42
    assert resultat["nombre_classes"] == 16
    assert set(resultat["candidats"]) == {
        "extra_trees_sans_competences",
        "svm_lineaire_sans_competences",
        "extra_trees_sans_moyenne",
    }


def test_validation_independante_ne_lit_pas_test(monkeypatch) -> None:
    lecture_originale = validation_independante.pd.read_csv
    chemins_lus = []

    def lire_csv(chemin, *args, **kwargs):
        chemins_lus.append(str(chemin))
        assert not str(chemin).endswith("test.csv")
        return lecture_originale(chemin, *args, **kwargs)

    monkeypatch.setattr(validation_independante.pd, "read_csv", lire_csv)
    executer_validation_independante(CHEMIN_DATASET)

    assert chemins_lus == [
        str(CHEMIN_DATASET / "train.csv"),
        str(CHEMIN_DATASET / "val.csv"),
    ]


def test_validation_independante_contient_les_metriques_demandees() -> None:
    resultat = executer_validation_independante(CHEMIN_DATASET)

    for candidat in resultat["candidats"].values():
        assert set(candidat["classes"]) == {
            f"parcours-{identifiant}"
            for identifiant in (
                "aee", "caa", "dtja", "emii", "emp", "esiia", "fic", "gca",
                "iaa", "icmp", "igglia", "imticia", "isaia", "pip", "tee", "teh",
            )
        }
        assert 0 <= candidat["f1_macro"] <= 1
        assert 0 <= candidat["balanced_accuracy"] <= 1
        assert 0 <= candidat["accuracy"] <= 1
        assert 0 <= candidat["precision_macro"] <= 1
        assert 0 <= candidat["rappel_macro"] <= 1
        assert 0 <= candidat["nombre_classes_rappel_nul"] <= 16
        assert len(candidat["matrice_confusion"]) == 16
        assert "rappel_par_classe" in candidat


def test_validation_independante_est_reproductible() -> None:
    premier = executer_validation_independante(CHEMIN_DATASET)
    second = executer_validation_independante(CHEMIN_DATASET)

    assert premier == second
