"""Index local de recherche documentaire basé sur TF-IDF."""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from orient_ia.rag.modeles import DocumentRAG

_STOP_WORDS = {
    "a",
    "acces",
    "afin",
    "alors",
    "apres",
    "au",
    "aucun",
    "aux",
    "avec",
    "besoin",
    "car",
    "ce",
    "ces",
    "comme",
    "comment",
    "dans",
    "de",
    "des",
    "deux",
    "donc",
    "du",
    "elle",
    "elles",
    "en",
    "est",
    "et",
    "etc",
    "etre",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "mais",
    "meme",
    "notamment",
    "ou",
    "par",
    "pas",
    "peut",
    "plus",
    "pour",
    "qu",
    "que",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "qui",
    "sa",
    "se",
    "ses",
    "sont",
    "sur",
    "tous",
    "tout",
    "une",
    "unes",
    "un",
    "vers",
}


class IndexRechercheDocuments:
    """Index documentaire local et déterministe basé sur TF-IDF."""

    def __init__(self, documents: Sequence[DocumentRAG]) -> None:
        self.documents = list(documents)
        self.vectoriseur = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        self.matrice = self.vectoriseur.fit_transform([document.contenu for document in self.documents])

    @staticmethod
    def _termes_importants(question: str) -> set[str]:
        mots = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", question.lower())
        return {mot for mot in mots if mot not in _STOP_WORDS and len(mot) > 2}

    def rechercher(
        self,
        question: str,
        nombre_resultats: int = 5,
        seuil: float = 0.05,
    ) -> list[dict[str, object]]:
        """Recherche les documents les plus pertinents pour une question."""
        if not question or not question.strip():
            return []

        termes = self._termes_importants(question)
        if not termes:
            return []

        contenus = [document.contenu.lower() for document in self.documents]
        if not any(terme in contenu for contenu in contenus for terme in termes):
            return []

        question_vector = self.vectoriseur.transform([question])
        scores = cosine_similarity(question_vector, self.matrice).ravel()
        indices = np.argsort(scores)[::-1]
        resultats: list[dict[str, object]] = []

        for indice in indices[: max(1, int(nombre_resultats))]:
            score = float(scores[int(indice)])
            if score < float(seuil):
                continue
            document = self.documents[int(indice)]
            resultats.append(document.vers_resultat(score))

        return resultats
