from datetime import date

import pytest

from orient_ia.domaine.modeles import (
    ResultatModeleML,
    ResultatRecommandation,
    Source,
    StatutSource,
)


def test_source_conserve_sa_provenance() -> None:
    source = Source(
        titre="Catalogue a verifier",
        origine="ISPM",
        url=None,
        date_consultation=date(2026, 8, 26),
        statut=StatutSource.INSTITUTIONNEL,
        donnees_extraites="A completer depuis une source fournie.",
    )

    assert source.statut == StatutSource.INSTITUTIONNEL
    assert source.date_consultation == date(2026, 8, 26)


def test_source_refuse_un_titre_vide() -> None:
    with pytest.raises(ValueError, match="titre"):
        Source(
            titre=" ",
            origine="ISPM",
            url=None,
            date_consultation=date(2026, 8, 26),
            statut=StatutSource.OFFICIEL,
            donnees_extraites="",
        )


def test_score_ml_est_borne() -> None:
    with pytest.raises(ValueError, match="compris entre 0 et 1"):
        ResultatModeleML(parcours="parcours-1", score=1.1, modele="baseline", version="0.1")


def test_recommandation_separe_les_sources_de_decision() -> None:
    resultat = ResultatRecommandation(
        resultats_ml=(ResultatModeleML("parcours-1", 0.8, "baseline", "0.1"),),
        informations_documentaires=("source-1#passage-1",),
        regles_pedagogiques=("regle-1",),
        incertitude="Le corpus est incomplet.",
    )

    assert resultat.resultats_ml[0].score == 0.8
    assert resultat.informations_documentaires != resultat.regles_pedagogiques