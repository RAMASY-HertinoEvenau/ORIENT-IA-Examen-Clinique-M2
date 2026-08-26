"""Prépare les matrices de features V2 en mémoire, sans entraîner de modèle."""
from __future__ import annotations

import json
from pathlib import Path

from orient_ia.ml.featurisation import (
    charger_splits,
    configurations_ablations,
    preparer_splits,
)

DOSSIER_DATASET = Path("data/full_sample_2000_v2")


def main() -> None:
    splits = charger_splits(DOSSIER_DATASET)
    identifiants_competences = tuple(json.loads(splits["train"].iloc[0]["competences"]).keys())
    resultats = {}
    for nom, configuration in configurations_ablations().items():
        matrices, cibles, featuriseur = preparer_splits(
            DOSSIER_DATASET,
            configuration,
            identifiants_competences,
        )
        resultats[nom] = {
            "train": matrices["train"].shape,
            "validation": matrices["val"].shape,
            "test": matrices["test"].shape,
            "cibles": {split: len(cible) for split, cible in cibles.items()},
            "nombre_features": len(featuriseur.get_feature_names_out()),
        }
    print(json.dumps(resultats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
