import csv
import json
from collections import Counter
from pathlib import Path

from orient_ia.generateur_dataset import GenerateurDataset


def _rows(paths: dict[str, Path]) -> list[dict[str, str]]:
    rows = []
    for split in ("train", "val", "test"):
        with paths[split].open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


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


def test_v2_cible_n_est_pas_reconstruite_par_la_regle_v1(tmp_path: Path):
    corpus = json.loads(Path("donnees/corpus_pedagogique.json").read_text(encoding="utf-8"))
    parcours = {item["identifiant"]: item for item in corpus["parcours"]}
    generator = GenerateurDataset(Path("donnees/corpus_pedagogique.json"))
    rows = _rows(generator.generer(n=400, seed=2026, out_dir=tmp_path / "v2"))

    reconstructions = 0
    for row in rows:
        competences = json.loads(row["competences"])
        scores = {}
        for parcours_id, definition in parcours.items():
            score = float(row["moyenne_scolaire"]) / 20.0
            related = definition.get("competences", [])
            if related:
                score += sum(competences.get(key, 0) for key in related) / (len(related) * 5) * 0.5
            scores[parcours_id] = score
        reconstructions += max(scores, key=scores.get) == row["parcours_cible"]

    assert reconstructions < len(rows)


def test_v2_couvre_les_parcours_sans_inventer_de_competences(tmp_path: Path):
    corpus_path = Path("donnees/corpus_pedagogique.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    generator = GenerateurDataset(corpus_path)
    rows = _rows(generator.generer(n=2000, seed=42, out_dir=tmp_path / "v2"))

    assert {row["parcours_cible"] for row in rows} == {
        item["identifiant"] for item in corpus["parcours"]
    }
    assert all(
        set(json.loads(row["competences"])) == set(generator.competences_reelles)
        for row in rows
    )
    assert Counter(row["parcours_cible"] for row in rows).most_common(1)[0][1] < 300


def test_v2_metadata_et_absence_de_donnees_personnelles(tmp_path: Path):
    generator = GenerateurDataset(Path("donnees/corpus_pedagogique.json"))
    paths = generator.generer(n=100, seed=7, out_dir=tmp_path / "v2")
    metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
    csv_content = "\n".join(
        paths[split].read_text(encoding="utf-8") for split in ("train", "val", "test")
    )
    columns = set(csv.DictReader(paths["train"].open(encoding="utf-8", newline="")).fieldnames or [])

    assert metadata["dataset_version"] == "v2"
    assert not columns.intersection({"nom", "prenom", "email", "telephone", "adresse"})
    assert "@" not in csv_content
    assert "telephone" not in csv_content.lower()
