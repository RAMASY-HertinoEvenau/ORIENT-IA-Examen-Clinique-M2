"""Moteur de recommandation ORIENT'IA fondé sur les artefacts ML gelés.

Le moteur charge le modèle final déjà sauvegardé et le featuriseur associé,
les combine aux informations vérifiées du corpus pédagogique, puis produit des
recommandations argumentées et prudentes sans jamais réentraîner le modèle.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from orient_ia.domaine.corpus import charger_corpus

_REPERTOIRE_PROJET = Path(__file__).resolve().parents[1]
_CHEMIN_MODELE = _REPERTOIRE_PROJET / "ml/modeles/extra_trees_sans_competences_run/modele.joblib"
_CHEMIN_FEATURISER = _REPERTOIRE_PROJET / "ml/modeles/extra_trees_sans_competences_run/featuriseur.joblib"
_CHEMIN_CORPUS = _REPERTOIRE_PROJET / "donnees/corpus_pedagogique.json"


def _normaliser_liste(valeur: Any) -> tuple[str, ...]:
    if valeur is None:
        return ()
    if isinstance(valeur, str):
        if not valeur.strip():
            return ()
        return tuple(part.strip() for part in valeur.split(";") if part.strip())
    if isinstance(valeur, (list, tuple, set)):
        return tuple(str(part).strip() for part in valeur if str(part).strip())
    return (str(valeur).strip(),)


def _normaliser_competences(valeur: Any) -> dict[str, int]:
    if valeur is None:
        return {}
    if isinstance(valeur, str):
        if not valeur.strip():
            return {}
        try:
            charge = json.loads(valeur)
        except json.JSONDecodeError:
            return {}
    else:
        charge = valeur
    if not isinstance(charge, dict):
        return {}
    resultats: dict[str, int] = {}
    for cle, note in charge.items():
        try:
            resultats[str(cle)] = int(note)
        except (TypeError, ValueError):
            continue
    return resultats


@dataclass(frozen=True)
class ProfilCandidat:
    """Profil candidat normalisé avant passage au moteur de recommandation."""

    identifiant: str | None = None
    matieres_preferees: tuple[str, ...] = ()
    moyenne_scolaire: float | None = None
    competences: dict[str, int] = field(default_factory=dict)
    centres_interet: tuple[str, ...] = ()
    projets: tuple[str, ...] = ()
    preferences_professionnelles: str | None = None
    environnement_travail: str | None = None

    @classmethod
    def depuis_dictionnaire(cls, donnees: Mapping[str, Any] | None) -> ProfilCandidat:
        if donnees is None:
            raise ValueError("Le profil candidat est obligatoire.")
        donnees = dict(donnees)
        moyenne = donnees.get("moyenne_scolaire")
        moyenne_float: float | None = None
        if moyenne is not None:
            try:
                moyenne_float = float(moyenne)
            except (TypeError, ValueError):
                moyenne_float = None

        return cls(
            identifiant=str(donnees.get("identifiant")) if donnees.get("identifiant") is not None else None,
            matieres_preferees=_normaliser_liste(donnees.get("matieres_preferees")),
            moyenne_scolaire=moyenne_float,
            competences=_normaliser_competences(donnees.get("competences")),
            centres_interet=_normaliser_liste(donnees.get("centres_interet")),
            projets=_normaliser_liste(donnees.get("projets")),
            preferences_professionnelles=(
                str(donnees.get("preferences_professionnelles")).strip()
                if donnees.get("preferences_professionnelles") not in (None, "")
                else None
            ),
            environnement_travail=(
                str(donnees.get("environnement_travail")).strip()
                if donnees.get("environnement_travail") not in (None, "")
                else None
            ),
        )

    def valider(self) -> list[str]:
        erreurs: list[str] = []
        if self.moyenne_scolaire is None:
            erreurs.append("moyenne_scolaire est obligatoire pour produire une recommandation.")
        elif not 0 <= self.moyenne_scolaire <= 20:
            erreurs.append("moyenne_scolaire doit être comprise entre 0 et 20.")
        if not self.competences:
            erreurs.append("au moins une compétence est nécessaire pour la recommandation.")
        return erreurs

    def vers_dataframe(self) -> pd.DataFrame:
        ligne = {
            "matieres_preferees": ";".join(self.matieres_preferees),
            "moyenne_scolaire": self.moyenne_scolaire,
            "competences": json.dumps(self.competences, ensure_ascii=False),
            "centres_interet": ";".join(self.centres_interet),
            "projets": ";".join(self.projets),
            "preferences_professionnelles": self.preferences_professionnelles or "",
            "environnement_travail": self.environnement_travail or "",
        }
        return pd.DataFrame([ligne])


class MoteurRecommandation:
    """Produit des recommandations argumentées à partir d'un profil candidat."""

    def __init__(
        self,
        chemin_modele: str | Path = _CHEMIN_MODELE,
        chemin_featuriseur: str | Path = _CHEMIN_FEATURISER,
        chemin_corpus: str | Path = _CHEMIN_CORPUS,
    ) -> None:
        self.chemin_modele = Path(chemin_modele)
        self.chemin_featuriseur = Path(chemin_featuriseur)
        self.chemin_corpus = Path(chemin_corpus)

        self.modele = joblib.load(self.chemin_modele)
        self.featuriseur = joblib.load(self.chemin_featuriseur)
        self.corpus = charger_corpus(self.chemin_corpus)
        self._parcours_par_identifiant = {parcours.identifiant: parcours for parcours in self.corpus.parcours}
        self._competences_par_identifiant = {
            competence.identifiant: competence for competence in self.corpus.competences
        }
        self._prerequis_par_identifiant = {
            prerequis.identifiant: prerequis for prerequis in self.corpus.prerequis
        }
        self._metiers_par_identifiant = {metier.identifiant: metier for metier in self.corpus.metiers}
        self._sources_par_identifiant = {source.identifiant: source for source in self.corpus.sources}
        self.corpus_identifiants = tuple(sorted(self._parcours_par_identifiant))

    def recommander(
        self,
        profil: Mapping[str, Any] | ProfilCandidat,
        nombre_resultats: int = 3,
        seuil_confiance: float = 0.15,
    ) -> dict[str, Any]:
        """Retourne une recommandation prudente avec sources et limites."""
        if isinstance(profil, ProfilCandidat):
            candidat = profil
        else:
            candidat = ProfilCandidat.depuis_dictionnaire(profil)

        erreurs = candidat.valider()
        if erreurs:
            return {
                "status": "incomplet",
                "message": "Profil incomplet : données insuffisantes pour produire une recommandation fiable.",
                "erreurs": erreurs,
                "recommandations": [],
                "incertitude": "Le profil candidat est incomplet et ne permet pas d'évaluer une recommandation avec prudence.",
                "profil": candidat.__dict__,
            }

        matrice = self.featuriseur.transform(candidat.vers_dataframe())
        probabilites = self.modele.predict_proba(matrice)[0]
        indices = np.argsort(probabilites)[::-1][: max(1, int(nombre_resultats))]
        recommandations: list[dict[str, Any]] = []
        meilleure_probabilite = float(probabilites[indices[0]])

        for index in indices:
            identifiant = str(self.modele.classes_[index])
            parcours = self._parcours_par_identifiant.get(identifiant)
            if parcours is None:
                continue
            score = float(probabilites[index])
            competences_parcours = [
                self._competences_par_identifiant[cid].nom for cid in parcours.competences if cid in self._competences_par_identifiant
            ]
            prerequis_parcours = [
                self._prerequis_par_identifiant[pid].description for pid in parcours.prerequis if pid in self._prerequis_par_identifiant
            ]
            metiers_parcours = [
                self._metiers_par_identifiant[mid].nom for mid in parcours.metiers if mid in self._metiers_par_identifiant
            ]

            raisons: list[str] = []
            if candidat.moyenne_scolaire is not None:
                raisons.append(f"Le profil indique une moyenne scolaire de {candidat.moyenne_scolaire:.1f}/20.")
            competences_pertinentes = [
                cid for cid in candidat.competences if cid in set(parcours.competences)
            ]
            if competences_pertinentes:
                raisons.append(
                    "Le profil marque des scores sur des compétences directement associées au parcours: "
                    + ", ".join(competences_pertinentes)
                    + "."
                )
            else:
                raisons.append(
                    "Le profil ne fournit pas de correspondance explicite avec les compétences institutionnelles du parcours."
                )
            if candidat.preferences_professionnelles:
                raisons.append(
                    f"Le profil indique une préférence professionnelle: {candidat.preferences_professionnelles}."
                )
            if candidat.environnement_travail:
                raisons.append(
                    f"Le profil indique une préférence d'environnement de travail: {candidat.environnement_travail}."
                )

            elements_du_corpus: list[str] = []
            if parcours.competences:
                elements_du_corpus.append(
                    "Compétences du parcours dans le corpus: "
                    + ", ".join(competences_parcours or [competence for competence in parcours.competences])
                    + "."
                )
            if parcours.prerequis:
                elements_du_corpus.append(
                    "Prérequis publiés: " + "; ".join(prerequis_parcours or parcours.prerequis) + "."
                )
            if parcours.metiers:
                elements_du_corpus.append(
                    "Débouchés ou métiers documentés dans le corpus: "
                    + ", ".join(metiers_parcours or parcours.metiers)
                    + "."
                )
            elements_du_corpus.append(
                f"Le parcours {parcours.nom} est référencé dans le corpus pédagogique vérifié."
            )

            sources = [
                self._sources_par_identifiant[source_id]
                for source_id in parcours.sources
                if source_id in self._sources_par_identifiant
            ]
            source_documentaire = [
                {
                    "titre": source.titre,
                    "url": source.url,
                    "origine": source.origine,
                    "section": source.section,
                }
                for source in sources
            ]
            limites = [
                "Cette recommandation ne constitue pas une décision officielle d'admission.",
                "Le corpus vérifié n'inclut pas l'ensemble des matières, passerelles ou débouchés de tous les parcours.",
            ]
            incertitude = "Le modèle est utilisé comme signal de pertinence, pas comme décision officielle."
            if score < seuil_confiance:
                incertitude += f" La probabilité estimée est faible ({score:.3f}), ce qui justifie une prudence accrue."

            recommandations.append(
                {
                    "formation": {
                        "identifiant": parcours.identifiant,
                        "nom": parcours.nom,
                    },
                    "score": round(score, 4),
                    "niveau_pertinence": self._niveau_pertinence(score),
                    "raisons_liees_au_profil": raisons,
                    "elements_du_corpus": elements_du_corpus,
                    "source_documentaire": source_documentaire,
                    "limites": limites,
                    "incertitude": incertitude,
                    "competences_du_parcours": list(parcours.competences),
                }
            )

        if not recommandations:
            return {
                "status": "incomplet",
                "message": "Aucune recommandation exploitable n'a pu être formée à partir du profil fourni.",
                "recommandations": [],
                "incertitude": "Aucune correspondance exploitable n'a été trouvée dans le corpus vérifié.",
                "profil": candidat.__dict__,
            }

        if meilleure_probabilite < seuil_confiance:
            status = "faible_confiance"
            message = "Le profil est exploitable mais la confiance du modèle est faible; la recommandation reste indicative."
        else:
            status = "ok"
            message = "Recommandation produite à partir du modèle final gelé et du corpus vérifié."

        return {
            "status": status,
            "message": message,
            "recommandations": recommandations,
            "incertitude": "Le système ne prend jamais une décision officielle d'admission; il fournit un signal de pertinence à interpréter avec prudence.",
            "profil": candidat.__dict__,
        }

    @staticmethod
    def _niveau_pertinence(score: float) -> str:
        if score >= 0.60:
            return "élevée"
        if score >= 0.35:
            return "moyenne"
        if score >= 0.20:
            return "faible"
        return "très faible"
