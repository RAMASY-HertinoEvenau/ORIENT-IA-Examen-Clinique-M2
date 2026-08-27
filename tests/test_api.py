from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from orient_ia.api import app

_VALID_PROFILE: dict[str, Any] = {
    "identifiant": "cand-100",
    "matieres_preferees": ["informatique", "mathematiques"],
    "moyenne_scolaire": 15.6,
    "competences": {
        "competence-techniques-informatiques-gestion": 4,
        "competence-electronique-systemes": 3,
    },
    "centres_interet": ["robotique"],
    "projets": ["ia"],
    "preferences_professionnelles": "salariat",
    "environnement_travail": "hybride",
}


client = TestClient(app)


def test_api_requete_valide_retourne_200() -> None:
    reponse = client.post("/api/recommandation", json={"profil": _VALID_PROFILE})

    assert reponse.status_code == 200
    body = reponse.json()
    assert "recommandations" in body
    assert "incertitude" in body
    assert "avertissements" in body
    assert "sources" in body


def test_api_profil_incomplet_retourne_erreur_explicite() -> None:
    reponse = client.post("/api/recommandation", json={"profil": {"identifiant": "cand-101"}})

    assert reponse.status_code == 400
    body = reponse.json()
    assert body["detail"]
    assert "profil" in body["detail"].lower() or "incomplet" in body["detail"].lower()


def test_api_format_json_invalide() -> None:
    reponse = client.post("/api/recommandation", data="{not valid json}", headers={"content-type": "application/json"})

    assert reponse.status_code == 422


def test_api_retourne_recommandation_avec_sources() -> None:
    reponse = client.post("/api/recommandation", json={"profil": _VALID_PROFILE})

    assert reponse.status_code == 200
    body = reponse.json()
    assert body["recommandations"]
    assert body["sources"]
    assert any(source.get("titre") for source in body["sources"])


def test_api_contient_avertissement_indicatif() -> None:
    reponse = client.post("/api/recommandation", json={"profil": _VALID_PROFILE})

    assert reponse.status_code == 200
    body = reponse.json()
    texte = " ".join(body["avertissements"]).lower()
    assert "recommandation indicative" in texte or "indicative" in texte or "decision officielle" in texte


def test_api_transforme_erreur_du_moteur() -> None:
    from orient_ia.api import service as service_module

    def _raise() -> dict[str, object]:
        raise RuntimeError("erreur métier simulée")

    service_module.RecommendationService().recommander = _raise  # type: ignore[assignment]

    reponse = client.post("/api/recommandation", json={"profil": _VALID_PROFILE})

    assert reponse.status_code == 500
    body = reponse.json()
    assert "detail" in body


def test_api_ne_fit_pas_ou_ne_reentraîne_pas(monkeypatch: pytest.MonkeyPatch) -> None:
    from orient_ia.ml.featurisation import FeaturiseurML

    def _lever_si_appelle(self, *args, **kwargs):
        raise AssertionError("L'API ne doit pas réentraîner le modèle.")

    monkeypatch.setattr(FeaturiseurML, "fit", _lever_si_appelle)
    reponse = client.post("/api/recommandation", json={"profil": _VALID_PROFILE})

    assert reponse.status_code == 200


def test_api_plusieurs_recommandations() -> None:
    reponse = client.post("/api/recommandation", json={"profil": _VALID_PROFILE})

    assert reponse.status_code == 200
    body = reponse.json()
    assert len(body["recommandations"]) >= 1
    assert body["recommandations"][0]["formation"]["identifiant"]
