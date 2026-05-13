from datetime import date
from pathlib import Path

import pytest

from orient_ia.domaine.corpus import charger_corpus
from orient_ia.domaine.modeles import (
    Competence,
    CorpusPedagogique,
    Formation,
    Matiere,
    Mention,
    Parcours,
    Passerelle,
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


def test_corpus_valide_les_relations_entre_entites() -> None:
    corpus = CorpusPedagogique(
        version="2026.08.26",
        mentions=(Mention("mention-1", "Sciences", ("parcours-1",)),),
        formations=(Formation("formation-1", "Master", ("mention-1",), "M2"),),
        parcours=(Parcours("parcours-1", "Data", ("matiere-1",), ("competence-1",)),),
        matieres=(Matiere("matiere-1", "Statistiques"),),
        competences=(Competence("competence-1", "Analyser"),),
    )

    corpus.valider()


def test_corpus_refuse_une_reference_inexistante() -> None:
    corpus = CorpusPedagogique(
        version="2026.08.26",
        mentions=(Mention("mention-1", "Sciences", ("parcours-inconnu",)),),
    )

    with pytest.raises(ValueError, match="Référence invalide"):
        corpus.valider()


def test_corpus_refuse_un_identifiant_duplique() -> None:
    corpus = CorpusPedagogique(
        version="2026.08.26",
        matieres=(Matiere("entite-1", "Statistiques"),),
        competences=(Competence("entite-1", "Analyser"),),
    )

    with pytest.raises(ValueError, match="Identifiant dupliqué"):
        corpus.valider()


def test_corpus_valide_une_passerelle_entre_parcours_existants() -> None:
    corpus = CorpusPedagogique(
        version="2026.08.26",
        parcours=(Parcours("parcours-1", "Data"), Parcours("parcours-2", "IA")),
        passerelles=(Passerelle("passerelle-1", "parcours-1", "parcours-2", "À documenter"),),
    )

    corpus.valider()


def test_corpus_versionne_contient_la_provenance_de_chaque_entite() -> None:
    corpus = charger_corpus(Path("donnees/corpus_pedagogique.json"))

    assert len(corpus.mentions) == 6
    assert len(corpus.parcours) == 16
    assert len(corpus.sources) == 4
    assert {source.statut for source in corpus.sources} == {
        StatutSource.INSTITUTIONNEL,
        StatutSource.EXTERNE,
    }
    assert all(entite.sources for entites in (
        corpus.mentions,
        corpus.formations,
        corpus.parcours,
        corpus.competences,
        corpus.prerequis,
        corpus.metiers,
    ) for entite in entites)
    assert not corpus.matieres
    assert not corpus.passerelles