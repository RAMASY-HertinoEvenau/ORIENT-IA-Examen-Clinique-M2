import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_script_no_fit_calls():
    src = Path("scripts/evaluer_test_final.py").read_text(encoding="utf-8")
    assert ".fit(" not in src and "fit_transform(" not in src, "Script must not call fit on test data"


def test_artifacts_and_metadata_present():
    base = Path("ml/modeles/extra_trees_sans_competences_run")
    assert (base / "modele.joblib").exists()
    assert (base / "featuriseur.joblib").exists()
    meta_p = base / "metadata.json"
    assert meta_p.exists()
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    # protocol checks (static)
    assert meta.get("modele") == "ExtraTreesClassifier"
    hp = meta.get("hyperparametres", {})
    assert int(hp.get("n_estimators", -1)) == 100
    assert int(hp.get("random_state", -1)) == 42
    assert int(hp.get("n_jobs", -1)) == 1
    assert meta.get("configuration") == "sans_competences"


@pytest.mark.skipif(os.environ.get("RUN_FINAL_TEST") != "1", reason="Final test evaluation disabled")
def test_run_evaluation_and_metrics():
    # This test intentionally runs the final evaluation. Enable by setting RUN_FINAL_TEST=1.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    subprocess.check_call([sys.executable, "scripts/evaluer_test_final.py"], env=env)

    report_json = Path("documentation/report_test_final.json")
    assert report_json.exists()
    data = json.loads(report_json.read_text(encoding="utf-8"))
    results = data.get("results", {})
    # Basic assertions on computed metrics
    assert "f1_macro" in results
    assert "confusion_matrix" in results
    cm = results["confusion_matrix"]
    assert len(cm) == 16 and len(cm[0]) == 16
