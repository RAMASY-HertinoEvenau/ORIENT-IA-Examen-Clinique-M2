from __future__ import annotations

from orient_ia.rag.chargeur import charger_documents_pedagogiques
from orient_ia.rag.index import IndexRechercheDocuments
from orient_ia.rag.service import rechercher_contexte, rechercher_documents


def test_chargement_du_corpus() -> None:
    documents = charger_documents_pedagogiques()

    assert documents
    assert all(document.contenu for document in documents)
    assert all(document.source for document in documents)


def test_indexation() -> None:
    documents = charger_documents_pedagogiques()
    index = IndexRechercheDocuments(documents)

    assert index.documents
    assert index.vectoriseur is not None
    assert index.matrice.shape[0] == len(documents)


def test_recherche_d_une_formation_connue() -> None:
    resultat = rechercher_documents("Qu'est-ce que l'IGGLIA ?", nombre_resultats=3)

    assert resultat["trouve"] is True
    assert resultat["resultats"]
    assert any("IGGLIA" in document["contenu"].upper() for document in resultat["resultats"])


def test_recherche_d_un_parcours_connu() -> None:
    resultat = rechercher_documents("Quel est le parcours ISAIA ?", nombre_resultats=3)

    assert resultat["trouve"] is True
    assert resultat["resultats"]
    assert any("ISAIA" in document["contenu"].upper() for document in resultat["resultats"])


def test_recherche_d_un_pre_requis() -> None:
    resultat = rechercher_documents("Quels sont les prérequis pour entrer à l'ISPM ?", nombre_resultats=3)

    assert resultat["trouve"] is True
    assert resultat["resultats"]
    assert any("baccalaureat" in document["contenu"].lower() for document in resultat["resultats"])


def test_recherche_avec_plusieurs_mots_cles() -> None:
    resultat = rechercher_documents("informatique intelligence artificielle admission dossier", nombre_resultats=3)

    assert resultat["trouve"] is True
    assert resultat["resultats"]


def test_presence_des_metadonnees_de_source() -> None:
    resultat = rechercher_documents("IGGLIA", nombre_resultats=1)

    document = resultat["resultats"][0]
    assert document["source"]
    assert document["url"] or document["source"]
    assert document["identifiant"]


def test_absence_de_resultat_hors_corpus() -> None:
    resultat = rechercher_documents("Planetes extraterrestres ou recettes de cuisine", nombre_resultats=3)

    assert resultat["trouve"] is False
    assert resultat["resultats"] == []
    assert "Aucune information suffis" in resultat["message"]


def test_respect_du_seuil_de_pertinence() -> None:
    resultat = rechercher_documents("IGGLIA", nombre_resultats=3, seuil=0.99)

    assert resultat["trouve"] is False
    assert resultat["resultats"] == []


def test_determinisme_des_resultats() -> None:
    premier = rechercher_documents("parcours ISAIA", nombre_resultats=3)
    second = rechercher_documents("parcours ISAIA", nombre_resultats=3)

    assert premier["trouve"] == second["trouve"]
    assert [item["identifiant"] for item in premier["resultats"]] == [
        item["identifiant"] for item in second["resultats"]
    ]


def test_contexte_agent() -> None:
    contexte = rechercher_contexte("Ques sont les parcours informatique ?", nombre_resultats=2)

    assert contexte["trouve"] is True
    assert contexte["contexte"]
    assert "IGGLIA" in contexte["contexte"].upper()
