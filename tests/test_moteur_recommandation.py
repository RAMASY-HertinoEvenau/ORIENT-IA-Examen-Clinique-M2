from __future__ import annotations

from pathlib import Path

import pytest

from orient_ia.domaine.corpus import charger_corpus
from orient_ia.ml.featurisation import FeaturiseurML
from orient_ia.moteur_recommandation import MoteurRecommandation


def _profil_valide() -> dict[str, object]:
    return {
        "identifiant": "cand-001",
        "matieres_preferees": ["mathematiques", "informatique"],
        "moyenne_scolaire": 15.8,
        "competences": {
            "competence-techniques-informatiques-gestion": 4,
            "competence-electronique-systemes": 3,
        },
        "centres_interet": ["robotique", "informatique"],
        "projets": ["ia", "programmation"],
        "preferences_professionnelles": "salariat",
        "environnement_travail": "hybride",
    }


def test_profil_valide_produit_une_recommandation() -> None:
    moteur = MoteurRecommandation()

    resultat = moteur.recommander(_profil_valide(), nombre_resultats=3)

    assert resultat["status"] in {"ok", "faible_confiance"}
    assert len(resultat["recommandations"]) >= 1
    assert resultat["recommandations"][0]["formation"]["identifiant"]


def test_profil_incomplet_est_rejete() -> None:
    moteur = MoteurRecommandation()
    profil = {
        "identifiant": "cand-002",
        "matieres_preferees": ["informatique"],
    }

    resultat = moteur.recommander(profil)

    assert resultat["status"] == "incomplet"
    assert resultat["erreurs"]
    assert "incomplet" in resultat["message"].lower()


def test_recommandation_croise_le_corpus_et_les_sources() -> None:
    moteur = MoteurRecommandation()
    resultat = moteur.recommander(_profil_valide(), nombre_resultats=2)

    assert resultat["recommandations"]
    for recommendation in resultat["recommandations"]:
        parcours_id = recommendation["formation"]["identifiant"]
        assert parcours_id in moteur.corpus_identifiants
        assert recommendation["elements_du_corpus"]
        assert recommendation["source_documentaire"]


def test_recommandation_si_donnees_insuffisantes() -> None:
    moteur = MoteurRecommandation()
    profil = {
        "identifiant": "cand-003",
        "moyenne_scolaire": 0.0,
        "competences": {},
        "matieres_preferees": [],
        "centres_interet": [],
        "projets": [],
        "preferences_professionnelles": "",
        "environnement_travail": "",
    }

    resultat = moteur.recommander(profil)

    assert resultat["status"] in {"incomplet", "faible_confiance"}
    assert "données" in resultat["message"].lower() or "faible" in resultat["message"].lower()


def test_moteur_indique_les_scores_faibles_ou_les_limites() -> None:
    moteur = MoteurRecommandation()
    profil = {
        "identifiant": "cand-004",
        "moyenne_scolaire": 8.0,
        "competences": {"competence-techniques-informatiques-gestion": 1},
        "matieres_preferees": [],
        "centres_interet": [] ,
        "projets": [],
        "preferences_professionnelles": "",
        "environnement_travail": "",
    }

    resultat = moteur.recommander(profil, seuil_confiance=0.95)

    assert resultat["status"] in {"ok", "faible_confiance", "incomplet"}
    assert resultat["incertitude"]


def test_moteur_n_invente_pas_d_information() -> None:
    moteur = MoteurRecommandation()
    corpus = charger_corpus(Path("donnees/corpus_pedagogique.json"))
    toutes_competences = {competence.identifiant for competence in corpus.competences}
    resultat = moteur.recommander(_profil_valide(), nombre_resultats=2)

    for recommendation in resultat["recommandations"]:
        for texte in recommendation["elements_du_corpus"]:
            assert "inconnu" not in texte.lower()
        for competence in recommendation.get("competences_du_parcours", []):
            assert competence in toutes_competences


def test_moteur_ne_fit_pas_au_cours_de_l_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    moteur = MoteurRecommandation()

    def _lever_si_appelle(self, *args, **kwargs):
        raise AssertionError("Le moteur ne doit pas réentraîner le modèle.")

    monkeypatch.setattr(FeaturiseurML, "fit", _lever_si_appelle)
    resultat = moteur.recommander(_profil_valide())

    assert resultat["status"] in {"ok", "faible_confiance"}
