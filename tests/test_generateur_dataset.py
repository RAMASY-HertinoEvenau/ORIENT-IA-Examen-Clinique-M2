import json
from pathlib import Path

from orient_ia.generateur_dataset import GenerateurDataset


def test_reproductibilite(tmp_path: Path):
    corpus = Path("donnees/corpus_pedagogique.json")
    g = GenerateurDataset(corpus)
    out1 = g.generer(n=100, seed=12345, out_dir=tmp_path / "run1")
    out2 = g.generer(n=100, seed=12345, out_dir=tmp_path / "run2")

    # same metadata
    m1 = json.loads((out1["meta"]).read_text(encoding="utf-8"))
    m2 = json.loads((out2["meta"]).read_text(encoding="utf-8"))
    assert m1["n_total"] == m2["n_total"] == 100
    assert m1["seed"] == m2["seed"] == 12345

    # files should be identical
    assert (out1["train"]).read_text(encoding="utf-8") == (out2["train"]).read_text(encoding="utf-8")
    assert (out1["val"]).read_text(encoding="utf-8") == (out2["val"]).read_text(encoding="utf-8")
    assert (out1["test"]).read_text(encoding="utf-8") == (out2["test"]).read_text(encoding="utf-8")


def test_validation_et_integrite(tmp_path: Path):
    corpus = Path("donnees/corpus_pedagogique.json")
    g = GenerateurDataset(corpus)
    out = g.generer(n=50, seed=1, out_dir=tmp_path / "sample")

    errors = g.valider_dataset(out)
    assert errors == [], f"Erreurs de validation détectées: {errors}"


def test_no_overlap_between_splits(tmp_path: Path):
    corpus = Path("donnees/corpus_pedagogique.json")
    g = GenerateurDataset(corpus)
    out = g.generer(n=60, seed=42, out_dir=tmp_path / "sample2")

    train_ids = set()
    for line in (out["train"]).read_text(encoding="utf-8").splitlines()[1:]:
        train_ids.add(line.split(",")[0])
    val_ids = set()
    for line in (out["val"]).read_text(encoding="utf-8").splitlines()[1:]:
        val_ids.add(line.split(",")[0])
    test_ids = set()
    for line in (out["test"]).read_text(encoding="utf-8").splitlines()[1:]:
        test_ids.add(line.split(",")[0])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
