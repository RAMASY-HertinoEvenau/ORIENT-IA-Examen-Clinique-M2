"""Évaluation finale sur `test.csv` selon le protocole gelé.

Usage (run once after manual checks):
  PYTHONPATH=. python scripts/evaluer_test_final.py

Le script écrit un rapport JSON et Markdown dans `documentation/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from orient_ia.ml.featurisation import COLONNE_CIBLE


def load_metadata(meta_p: Path) -> dict[str, Any]:
    return json.loads(meta_p.read_text(encoding="utf-8"))


def check_protocol(meta: dict[str, Any]) -> None:
    # Validate required frozen protocol fields
    if meta.get("modele") != "ExtraTreesClassifier":
        raise RuntimeError("Artifact model mismatch: expected ExtraTreesClassifier")
    hp = meta.get("hyperparametres", {})
    if int(hp.get("n_estimators", -1)) != 100:
        raise RuntimeError("n_estimators mismatch: expected 100")
    if int(hp.get("random_state", -1)) != 42:
        raise RuntimeError("random_state mismatch: expected 42")
    if int(hp.get("n_jobs", -1)) != 1:
        raise RuntimeError("n_jobs mismatch: expected 1")
    if meta.get("configuration") != "sans_competences":
        raise RuntimeError("configuration mismatch: expected 'sans_competences'")
    used = meta.get("used_files", [])
    if not any("train.csv" in p for p in used) or not any("val.csv" in p for p in used):
        raise RuntimeError("metadata does not indicate training on train+val")


def evaluate(modele_p: Path, featuriseur_p: Path, test_p: Path, meta: dict[str, Any]) -> dict[str, Any]:
    modele = joblib.load(modele_p)
    featuriseur = joblib.load(featuriseur_p)

    # Safety: do not call fit/fit_transform on featuriseur
    test_df = pd.read_csv(test_p)
    X_df = test_df.drop(columns=[COLONNE_CIBLE], errors="ignore")
    # Remove identifier columns that are not valid features for the featuriser
    if "id_candidat" in X_df.columns:
        X_df = X_df.drop(columns=["id_candidat"])
    y = test_df[COLONNE_CIBLE] if COLONNE_CIBLE in test_df.columns else None

    X = featuriseur.transform(X_df)

    has_proba = hasattr(modele, "predict_proba")
    if has_proba:
        y_proba = modele.predict_proba(X)
    else:
        y_proba = None

    y_pred = modele.predict(X)

    labels = meta.get("classes") or sorted(np.unique(np.concatenate([y_pred, y])) if y is not None else np.unique(y_pred))

    results: dict[str, Any] = {}
    results["n_test_examples"] = len(test_df)
    results["n_classes"] = len(labels)
    results["classes"] = labels
    results["seed"] = int(meta.get("seed", 0))
    results["model"] = meta.get("modele")
    results["configuration"] = meta.get("configuration")
    results["hyperparametres"] = meta.get("hyperparametres")
    results["artefacts"] = meta.get("artefacts")

    if y is None:
        raise RuntimeError("test.csv does not contain target column 'parcours'")

    results["f1_macro"] = float(f1_score(y, y_pred, average="macro", labels=labels))
    results["balanced_accuracy"] = float(balanced_accuracy_score(y, y_pred))
    results["accuracy"] = float(accuracy_score(y, y_pred))
    results["precision_macro"] = float(precision_score(y, y_pred, average="macro", zero_division=0, labels=labels))
    results["recall_macro"] = float(recall_score(y, y_pred, average="macro", zero_division=0, labels=labels))

    recall_per_class = recall_score(y, y_pred, average=None, labels=labels, zero_division=0)
    results["recall_per_class"] = {lab: float(r) for lab, r in zip(labels, recall_per_class)}
    results["n_classes_recall_zero"] = int(sum(1 for r in recall_per_class if r == 0.0))

    cm = confusion_matrix(y, y_pred, labels=labels)
    results["confusion_matrix_shape"] = [int(cm.shape[0]), int(cm.shape[1])]
    results["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        try:
            results["log_loss"] = float(log_loss(y, y_proba, labels=labels))
        except (ValueError, TypeError):
            results["log_loss"] = None
    else:
        results["log_loss"] = None

    return results


def write_reports(results: dict[str, Any], out_md: Path, out_json: Path, meta: dict[str, Any]) -> None:
    out_json.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Rapport final — évaluation sur test.csv\n")
    lines.append("## Protocole gelé\n")
    lines.append("Voir documentation/protocole-gel.md pour les règles du protocole.\n")
    lines.append("## Artefacts utilisés\n")
    lines.append(f"- modèle: {meta.get('modele')}\n")
    lines.append(f"- configuration: {meta.get('configuration')}\n")
    lines.append(f"- hyperparametres: {meta.get('hyperparametres')}\n")
    lines.append(f"- seed: {meta.get('seed')}\n")
    lines.append(f"- artefacts: {meta.get('artefacts')}\n")
    lines.append("## Résultats\n")
    lines.append(f"- nombre d'exemples: {results.get('n_test_examples')}\n")
    lines.append(f"- nombre de classes: {results.get('n_classes')}\n")
    lines.append(f"- classes: {results.get('classes')}\n")
    lines.append(f"- F1 macro: {results.get('f1_macro')}\n")
    lines.append(f"- balanced accuracy: {results.get('balanced_accuracy')}\n")
    lines.append(f"- accuracy: {results.get('accuracy')}\n")
    lines.append(f"- precision macro: {results.get('precision_macro')}\n")
    lines.append(f"- recall macro: {results.get('recall_macro')}\n")
    lines.append(f"- nombre de classes avec rappel nul: {results.get('n_classes_recall_zero')}\n")
    lines.append("\n## Rappels par classe\n")
    for c, r in results.get("recall_per_class", {}).items():
        lines.append(f"- {c}: {r}\n")
    lines.append("\n## Matrice de confusion\n")
    lines.append("```")
    for row in results.get("confusion_matrix", []):
        lines.append(" ".join(str(int(x)) for x in row))
    lines.append("```")
    lines.append("\n## Limites et interprétation prudente\n")
    lines.append("Les données sont synthétiques; éviter toute extrapolation vers des populations réelles.\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    base = Path("ml/modeles/extra_trees_sans_competences_run")
    modele_p = base / "modele.joblib"
    featuriseur_p = base / "featuriseur.joblib"
    meta_p = base / "metadata.json"

    for p in (modele_p, featuriseur_p, meta_p):
        if not p.exists():
            raise FileNotFoundError(f"Artefact manquant: {p}")

    meta = load_metadata(meta_p)
    check_protocol(meta)

    test_p = Path("data/full_sample_2000_v2/test.csv")
    if not test_p.exists():
        raise FileNotFoundError("Fichier de test introuvable: data/full_sample_2000_v2/test.csv")

    results = evaluate(modele_p, featuriseur_p, test_p, meta)

    out_json = Path("documentation/report_test_final.json")
    out_md = Path("documentation/phase-2-3-evaluation-finale-test.md")
    write_reports(results, out_md, out_json, meta)

    print(json.dumps({"status": "ok", "report": str(out_md), "json": str(out_json)}))


if __name__ == "__main__":
    main()
