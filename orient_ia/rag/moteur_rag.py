"""Moteur RAG pour la recherche documentaire avec indexation hybride et traçabilité des sources."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normaliser_texte(texte: str) -> str:
    """Normalise un texte (suppression des accents, minuscules, ponctuation)."""
    texte = unicodedata.normalize("NFKD", texte).encode("ASCII", "ignore").decode("utf-8")
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", " ", texte)
    return " ".join(texte.split())


@dataclass
class SourceReference:
    identifiant: str
    titre: str
    origine: str
    url: str
    date_consultation: str
    statut: str
    section: str
    donnees_extraites: str
    limites: str
    incertitudes: str


@dataclass
class PassageDocumentaire:
    identifiant: str
    titre: str
    categorie: str
    contenu: str
    sources_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultatRecherche:
    passage: PassageDocumentaire
    score: float
    sources: List[SourceReference]


class MoteurRAG:
    """Moteur RAG pour l'indexation et la recherche dans le corpus pédagogique ISPM."""

    def __init__(self, chemin_corpus: str | Path = "donnees/corpus_pedagogique.json"):
        self.chemin_corpus = Path(chemin_corpus)
        self.sources_registre: Dict[str, SourceReference] = {}
        self.passages: List[PassageDocumentaire] = []
        self.vectoriseur: Optional[TfidfVectorizer] = None
        self.matrice_tfidf: Optional[np.ndarray] = None
        self.raw_data: Dict[str, Any] = {}
        self._charger_corpus()
        self._indexer()

    def _charger_corpus(self) -> None:
        if not self.chemin_corpus.exists():
            raise FileNotFoundError(f"Corpus introuvable : {self.chemin_corpus}")

        with self.chemin_corpus.open("r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        # Charger le registre des sources
        for s in self.raw_data.get("sources", []):
            self.sources_registre[s["identifiant"]] = SourceReference(
                identifiant=s["identifiant"],
                titre=s.get("titre", ""),
                origine=s.get("origine", ""),
                url=s.get("url", ""),
                date_consultation=s.get("date_consultation", ""),
                statut=s.get("statut", "institutionnel"),
                section=s.get("section", ""),
                donnees_extraites=s.get("donnees_extraites", ""),
                limites=s.get("limites", ""),
                incertitudes=s.get("incertitudes", ""),
            )

        # 1. Découpage : Mentions et formations
        for m in self.raw_data.get("mentions", []):
            nom_m = m["nom"]
            parcours_list = m.get("parcours", [])
            contenu = f"Mention : {nom_m}. Cette mention regroupe les parcours suivants : {', '.join(parcours_list)}."
            self.passages.append(
                PassageDocumentaire(
                    identifiant=m["identifiant"],
                    titre=f"Mention {nom_m}",
                    categorie="mention",
                    contenu=contenu,
                    sources_ids=m.get("sources", []),
                    metadata={"type": "mention", "nom": nom_m, "parcours": parcours_list},
                )
            )

        # 2. Découpage : Parcours détaillés
        competences_map = {c["identifiant"]: c for c in self.raw_data.get("competences", [])}
        prerequis_map = {p["identifiant"]: p for p in self.raw_data.get("prerequis", [])}
        metiers_map = {m["identifiant"]: m for m in self.raw_data.get("metiers", [])}
        matieres_map = {mat["identifiant"]: mat for mat in self.raw_data.get("matieres", [])}

        for p in self.raw_data.get("parcours", []):
            id_p = p["identifiant"]
            nom_p = p["nom"]
            comps = [competences_map[c_id]["nom"] for c_id in p.get("competences", []) if c_id in competences_map]
            prereqs = [prerequis_map[pr_id]["description"] for pr_id in p.get("prerequis", []) if pr_id in prerequis_map]
            mets = [metiers_map[m_id]["nom"] for m_id in p.get("metiers", []) if m_id in metiers_map]
            mats = [matieres_map[mat_id]["nom"] for mat_id in p.get("matieres", []) if mat_id in matieres_map]

            comps_str = "; ".join(comps) if comps else "Compétences professionnelles fondamentales du département."
            prereqs_str = " ".join(prereqs) if prereqs else "Sélection sur dossier BACC."
            mets_str = "; ".join(mets) if mets else "Débouchés professionnels du secteur d'activité."
            mats_str = "; ".join(mats) if mats else "Enseignements fondamentaux et de spécialité."

            contenu = (
                f"Parcours {nom_p} ({id_p}). "
                f"Compétences visées : {comps_str}. "
                f"Matières principales : {mats_str}. "
                f"Conditions d'admission et prérequis : {prereqs_str}. "
                f"Débouchés professionnels répertoriés : {mets_str}."
            )

            self.passages.append(
                PassageDocumentaire(
                    identifiant=id_p,
                    titre=f"Parcours {nom_p}",
                    categorie="parcours",
                    contenu=contenu,
                    sources_ids=p.get("sources", []),
                    metadata={
                        "type": "parcours",
                        "id": id_p,
                        "nom": nom_p,
                        "competences": comps,
                        "matieres": mats,
                        "prerequis": prereqs,
                        "metiers": mets,
                    },
                )
            )

        # 3. Découpage : Conditions d'accès et Prérequis généraux
        for pr in self.raw_data.get("prerequis", []):
            self.passages.append(
                PassageDocumentaire(
                    identifiant=pr["identifiant"],
                    titre=f"Prérequis : {pr['identifiant']}",
                    categorie="prerequis",
                    contenu=f"Règle de prérequis ISPM : {pr['description']}",
                    sources_ids=pr.get("sources", []),
                    metadata={"type": "prerequis", "obligatoire": pr.get("obligatoire", True)},
                )
            )

        # 4. Découpage : Contradictions et limites
        for c in self.raw_data.get("contradictions", []):
            self.passages.append(
                PassageDocumentaire(
                    identifiant=f"contradiction-{c['sujet']}",
                    titre=f"Contradiction : {c['sujet']}",
                    categorie="contradiction",
                    contenu=f"Sujet : {c['sujet']}. Description : {c['description']} Décision retenue : {c['decision']}",
                    sources_ids=[c.get("source_a", ""), c.get("source_b", "")],
                    metadata={"type": "contradiction", "sujet": c["sujet"]},
                )
            )

        # 5. Découpage : Informations absentes
        for info in self.raw_data.get("informations_absentes", []):
            self.passages.append(
                PassageDocumentaire(
                    identifiant=f"info-absente-{_normaliser_texte(info)}",
                    titre=f"Information absente : {info}",
                    categorie="absence",
                    contenu=f"Information non disponible dans le corpus officiel ISPM : {info}.",
                    sources_ids=["ispm-filieres", "ispm-presentation"],
                    metadata={"type": "absence", "intitule": info},
                )
            )

    def _indexer(self) -> None:
        """Crée l'index TF-IDF pour la recherche vectorielle lexicale."""
        corpus_textes = [_normaliser_texte(p.titre + " " + p.contenu) for p in self.passages]
        self.vectoriseur = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrice_tfidf = self.vectoriseur.fit_transform(corpus_textes)

    def rechercher(self, requete: str, top_k: int = 4, seuil_min: float = 0.05) -> List[ResultatRecherche]:
        """Effectue une recherche hybride (lexicale + vectorielle) et retourne les passages classés."""
        if not requete.strip() or self.vectoriseur is None or self.matrice_tfidf is None:
            return []

        req_norm = _normaliser_texte(requete)
        vect_req = self.vectoriseur.transform([req_norm])
        scores_cosinus = cosine_similarity(vect_req, self.matrice_tfidf)[0]

        # Scoring hybride avec bonus pour correspondance exacte de mots clés signifiants (acronymes, mentions)
        mots_vides = {"de", "du", "des", "le", "la", "les", "un", "une", "en", "et", "a", "au", "aux", "pour", "dans", "par", "sur", "vous", "proposez", "propose", "est", "ce", "qui", "que", "avec", "sans", "faire", "quel", "quelle", "quels", "quelles"}
        mots_cles = {m for m in req_norm.split() if m not in mots_vides and len(m) > 2}
        resultats: List[Tuple[float, PassageDocumentaire]] = []

        for idx, passage in enumerate(self.passages):
            score_base = float(scores_cosinus[idx])
            texte_passage_norm = _normaliser_texte(passage.titre + " " + passage.contenu)
            mots_passage = set(texte_passage_norm.split())

            # Bonus de correspondance lexicale sur les termes signifiants
            intersection = mots_cles.intersection(mots_passage)
            bonus_mots = len(intersection) * 0.15

            score_final = score_base + bonus_mots
            if score_final >= seuil_min:
                resultats.append((score_final, passage))

        # Trier par score décroissant
        resultats.sort(key=lambda x: x[0], reverse=True)
        top_resultats = resultats[:top_k]

        sorties: List[ResultatRecherche] = []
        for score, passage in top_resultats:
            sources_obj = [self.sources_registre[s_id] for s_id in passage.sources_ids if s_id in self.sources_registre]
            sorties.append(ResultatRecherche(passage=passage, score=round(score, 4), sources=sources_obj))

        return sorties

    def verifier_presence_formation(self, nom_formation: str) -> bool:
        """Vérifie si une filière ou un parcours existe officiellement dans le corpus ISPM."""
        nom_norm = _normaliser_texte(nom_formation)
        for p in self.passages:
            if p.categorie in ("parcours", "mention"):
                if nom_norm in _normaliser_texte(p.titre) or nom_norm in _normaliser_texte(p.metadata.get("nom", "")):
                    return True
        return False

    def get_sources_parcours(self, id_parcours: str) -> List[SourceReference]:
        """Retourne la liste des sources associées à un identifiant de parcours."""
        for p in self.passages:
            if p.identifiant == id_parcours or p.metadata.get("id") == id_parcours:
                return [self.sources_registre[s_id] for s_id in p.sources_ids if s_id in self.sources_registre]
        return []
