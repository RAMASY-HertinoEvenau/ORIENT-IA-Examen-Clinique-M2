import json
from pathlib import Path

import numpy as np
import pytest

from orient_ia.ml.featurisation import (
    COLONNES_ENTREE,
    ConfigurationFeatures,
    FeaturiseurML,
    charger_splits,
    configurations_ablations,
    preparer_splits,
    separer_cible,
)

CHEMIN_DATASET = Path("data/full_sample_2000_v2")


def _identifiants_competences() -> tuple[str, ...]:
    splits = charger_splits(CHEMIN_DATASET)
    return tuple(json.loads(splits["train"].iloc[0]["competences"]).keys())


def test_separation_exclut_cible_et_identifiant() -> None:
    train = charger_splits(CHEMIN_DATASET)["train"]
    entrees, cible = separer_cible(train)

    assert "parcours_cible" not in entrees
    assert "id_candidat" not in entrees
    assert tuple(entrees.columns) == COLONNES_ENTREE
    assert len(entrees) == len(cible)


def test_preparation_des_trois_splits_sans_entrainement() -> None:
    matrices, cibles, featuriseur = preparer_splits(CHEMIN_DATASET)

    assert {nom: matrice.shape[0] for nom, matrice in matrices.items()} == {
        "train": 1400,
        "val": 300,
        "test": 300,
    }
    assert {nom: len(cible) for nom, cible in cibles.items()} == {
        "train": 1400,
        "val": 300,
        "test": 300,
    }
    noms = set(featuriseur.get_feature_names_out())
    assert all("parcours_cible" not in nom for nom in noms)
    assert all("id_candidat" not in nom for nom in noms)


def test_projets_vides_deviennent_un_token_explice() -> None:
    train = charger_splits(CHEMIN_DATASET)["train"]
    entrees, _ = separer_cible(train)
    featuriseur = FeaturiseurML(_identifiants_competences())
    featuriseur.fit(entrees)

    noms = set(featuriseur.get_feature_names_out())
    assert any("aucun_projet_declare" in nom for nom in noms)


def test_vocabulaire_et_categories_sont_appris_sur_train_uniquement() -> None:
    train = charger_splits(CHEMIN_DATASET)["train"].head(20).copy()
    validation = train.head(1).copy()
    validation.loc[:, "matieres_preferees"] = "modalite_absente_validation"
    validation.loc[:, "preferences_professionnelles"] = "modalite_absente_validation"
    validation.loc[:, "projets"] = "projet_absent_validation"
    entrees_train, _ = separer_cible(train)
    entrees_validation, _ = separer_cible(validation)
    featuriseur = FeaturiseurML(_identifiants_competences())

    featuriseur.fit(entrees_train)
    matrice_validation = featuriseur.transform(entrees_validation)

    assert matrice_validation.shape[0] == 1
    assert "modalite_absente_validation" not in featuriseur.get_feature_names_out()
    assert "projet_absent_validation" not in featuriseur.get_feature_names_out()


def test_changer_la_cible_ne_change_pas_les_features() -> None:
    train = charger_splits(CHEMIN_DATASET)["train"].head(50).copy()
    train_modifie = train.copy()
    train_modifie["parcours_cible"] = "cible_modifiee"
    entrees, _ = separer_cible(train)
    entrees_modifiees, _ = separer_cible(train_modifie)
    featuriseur = FeaturiseurML(_identifiants_competences())

    premiere_matrice = featuriseur.fit_transform(entrees)
    seconde_matrice = featuriseur.transform(entrees_modifiees)

    assert np.array_equal(premiere_matrice.toarray(), seconde_matrice.toarray())


@pytest.mark.parametrize("nom_configuration", list(configurations_ablations()))
def test_toutes_les_ablations_sont_executables(nom_configuration: str) -> None:
    configuration = configurations_ablations()[nom_configuration]
    matrices, cibles, featuriseur = preparer_splits(
        CHEMIN_DATASET,
        configuration,
        _identifiants_competences(),
    )

    assert matrices["train"].shape[0] == len(cibles["train"]) == 1400
    noms = set(featuriseur.get_feature_names_out())
    assert "parcours_cible" not in noms
    assert "id_candidat" not in noms
    if not configuration.utiliser_moyenne:
        assert not any("moyenne" in nom for nom in noms)
    if not configuration.utiliser_competences:
        assert not any("competence_" in nom for nom in noms)
    if not configuration.utiliser_variables_textuelles:
        assert not any(nom.startswith("texte_") for nom in noms)


def test_configuration_sans_feature_refusee() -> None:
    configuration = ConfigurationFeatures(
        utiliser_moyenne=False,
        utiliser_competences=False,
        utiliser_variables_textuelles=False,
        utiliser_variables_synthetiques=False,
    )
    with pytest.raises(ValueError, match="aucune feature"):
        preparer_splits(CHEMIN_DATASET, configuration, _identifiants_competences())
