"""Entraînement final strict selon le protocole gelé.

Ne lit que `train.csv` et `val.csv`, n'accède jamais à `test.csv`.
Sauvegarde featuriseur, modèle et metadata dans un répertoire d'artefacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from orient_ia.ml.experimentations import _identifiants_competences
from orient_ia.ml.featurisation import FeaturiseurML, configurations_ablations, separer_cible


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dossier", type=str, default="data/full_sample_2000_v2")
    p.add_argument("--out", type=str, default="ml/modeles/extra_trees_sans_competences")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dossier = Path(args.dossier)
    train_p = dossier / "train.csv"
    val_p = dossier / "val.csv"

    if not train_p.exists() or not val_p.exists():
        raise FileNotFoundError("train.csv ou val.csv introuvable dans le dossier spécifié")

    checksum_train_before = md5(train_p)
    checksum_val_before = md5(val_p)

    train = pd.read_csv(train_p)
    val = pd.read_csv(val_p)

    # Identifiants de compétences pris depuis le train (si disponibles)
    identifiants = _identifiants_competences(train)

    config = configurations_ablations()["sans_competences"]

    # Concat train + val pour entraînement final
    concat = pd.concat([train, val], ignore_index=True)
    entrees, cible = separer_cible(concat)

    featuriseur = FeaturiseurML(identifiants, config)
    matrice = featuriseur.fit_transform(entrees)

    # Entraînement du modèle fixé
    modele = ExtraTreesClassifier(n_estimators=100, random_state=args.seed, n_jobs=1)
    modele.fit(matrice, cible)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    modele_p = out_dir / "modele.joblib"
    featuriseur_p = out_dir / "featuriseur.joblib"
    meta_p = out_dir / "metadata.json"

    joblib.dump(modele, modele_p)
    joblib.dump(featuriseur, featuriseur_p)

    metadata = {
        "protocole": "gel_extra_trees_sans_competences",
        "modele": "ExtraTreesClassifier",
        "hyperparametres": {"n_estimators": 100, "random_state": args.seed, "n_jobs": 1},
        "configuration": "sans_competences",
        "seed": args.seed,
        "used_files": [str(train_p), str(val_p)],
        "checksums": {"train.csv": checksum_train_before, "val.csv": checksum_val_before},
        "artefacts": {"modele": str(modele_p), "featuriseur": str(featuriseur_p)},
        "classes": sorted(cible.unique()),
        "nombre_exemples": len(concat),
    }

    with meta_p.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(json.dumps({"status": "ok", "out": str(out_dir), "n": len(concat)}))


if __name__ == "__main__":
    main()
