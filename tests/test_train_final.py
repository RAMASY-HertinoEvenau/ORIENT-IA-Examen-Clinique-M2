import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from joblib import load

from orient_ia.ml.featurisation import separer_cible


def test_train_final_cree_artefacts(tmp_path: Path):
    out = tmp_path / "artefacts"
    # Run the script
    cmd = [sys.executable, "scripts/train_final.py", "--out", str(out), "--seed", "42"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    subprocess.check_call(cmd, env=env)

    modele_p = out / "modele.joblib"
    featuriseur_p = out / "featuriseur.joblib"
    meta_p = out / "metadata.json"

    assert modele_p.exists()
    assert featuriseur_p.exists()
    assert meta_p.exists()

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    # used_files doit contenir uniquement train et val
    assert all("train.csv" in f or "val.csv" in f for f in meta["used_files"]) 
    assert "test.csv" not in json.dumps(meta)


def test_train_final_reproductible(tmp_path: Path):
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    # subprocess and sys are imported at module level
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    subprocess.check_call([sys.executable, "scripts/train_final.py", "--out", str(out1), "--seed", "42"], env=env)
    subprocess.check_call([sys.executable, "scripts/train_final.py", "--out", str(out2), "--seed", "42"], env=env)
    # Load artefacts and compare predictions on a fixed sample to ensure reproducibility
    model1 = load(out1 / "modele.joblib")
    feat1 = load(out1 / "featuriseur.joblib")
    model2 = load(out2 / "modele.joblib")
    feat2 = load(out2 / "featuriseur.joblib")

    dossier = Path("data/full_sample_2000_v2")
    train = pd.read_csv(dossier / "train.csv")
    val = pd.read_csv(dossier / "val.csv")
    concat = pd.concat([train, val], ignore_index=True)
    # use first 50 examples as deterministic test sample
    sample = concat.iloc[:50]
    entrees, _ = separer_cible(sample)
    X1 = feat1.transform(entrees)
    X2 = feat2.transform(entrees)
    p1 = model1.predict(X1)
    p2 = model2.predict(X2)
    assert (p1 == p2).all()
