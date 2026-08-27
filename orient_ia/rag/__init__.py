"""Module RAG pour la recherche documentaire, l'indexation et la traçabilité des sources ORIENT'IA."""
from orient_ia.rag.chargeur import charger_documents_pedagogiques
from orient_ia.rag.index import IndexRechercheDocuments
from orient_ia.rag.moteur_rag import MoteurRAG, PassageDocumentaire, ResultatRecherche
from orient_ia.rag.service import rechercher_contexte, rechercher_documents

__all__ = [
    "IndexRechercheDocuments",
    "charger_documents_pedagogiques",
    "rechercher_contexte",
    "rechercher_documents",
    "MoteurRAG",
    "PassageDocumentaire",
    "ResultatRecherche",
]
