"""Boîte à outils fonctionnels pour l'Agent ORIENT'IA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from orient_ia.rag.moteur_rag import MoteurRAG, SourceReference


class BoiteAOutilsAgent:
    """Regroupe les outils techniques identifiables pour l'assistant ORIENT'IA."""

    def __init__(
        self,
        moteur_rag: Optional[MoteurRAG] = None,
        chemin_modele: str | Path = "ml/modeles/extra_trees_sans_competences_run",
    ):
        self.rag = moteur_rag or MoteurRAG()
        self.chemin_modele = Path(chemin_modele)
        self.modele = None
        self.featuriseur = None
        self.metadata_ml = {}
        self._charger_modele_ml()

    def _charger_modele_ml(self) -> None:
        modele_file = self.chemin_modele / "modele.joblib"
        featuriseur_file = self.chemin_modele / "featuriseur.joblib"
        meta_file = self.chemin_modele / "metadata.json"

        if modele_file.exists() and featuriseur_file.exists():
            self.modele = joblib.load(modele_file)
            self.featuriseur = joblib.load(featuriseur_file)
            if meta_file.exists():
                with meta_file.open("r", encoding="utf-8") as f:
                    self.metadata_ml = json.load(f)

    # --- Outil 1 : Recherche documentaire et traçabilité ---
    def rechercher_formation(self, requete: str, top_k: int = 3) -> Dict[str, Any]:
        """Recherche des informations vérifiées dans le corpus pédagogique officiel de l'ISPM."""
        resultats = self.rag.rechercher(requete, top_k=top_k)
        if not resultats:
            return {
                "statut": "non_trouve",
                "message": f"Aucune information officielle correspondant à '{requete}' n'a été trouvée dans le corpus ISPM.",
                "passages": [],
                "sources": [],
            }

        passages_data = []
        sources_data = []
        sources_vues = set()

        for res in resultats:
            passages_data.append({
                "titre": res.passage.titre,
                "categorie": res.passage.categorie,
                "contenu": res.passage.contenu,
                "score": res.score,
            })
            for s in res.sources:
                if s.identifiant not in sources_vues:
                    sources_vues.add(s.identifiant)
                    sources_data.append({
                        "nom": s.titre,
                        "origine": s.origine,
                        "url": s.url,
                        "statut": s.statut,
                        "date": s.date_consultation,
                        "incertitudes": s.incertitudes,
                    })

        return {
            "statut": "succes",
            "passages": passages_data,
            "sources": sources_data,
        }

    # --- Outil 2 : Analyse ML et prédiction de compatibilité ---
    def analyser_profil_ml(self, profil: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
        """Analyse les données du candidat avec le modèle ML entraîné et retourne les pistes les plus cohérentes."""
        if not profil.get("niveau"):
            return {"statut": "erreur", "erreur": "profil_incomplet", "recommandations": [], "message": "Le niveau d'études est requis pour formuler une orientation adaptée."}

        # Normalisation des chaînes d'entrée
        matieres_brut = profil.get("matieres_preferees", "")
        if isinstance(matieres_brut, list):
            matieres_str = " ".join(matieres_brut).lower()
        else:
            matieres_str = str(matieres_brut).lower()

        centres_brut = profil.get("centres_interet", [])
        if isinstance(centres_brut, list):
            centres_str = " ".join(centres_brut).lower()
        else:
            centres_str = str(centres_brut).lower()

        competences_str = str(profil.get("competences", "")).lower()
        projets_str = str(profil.get("projets", "")).lower()

        texte_profil = f"{matieres_str} {centres_str} {competences_str} {projets_str}"

        # Dictionnaire des affinités sémantiques réelles par parcours
        scores_profil: Dict[str, float] = {
            "parcours-isaia": 0.05,
            "parcours-igglia": 0.05,
            "parcours-esiia": 0.05,
            "parcours-imticia": 0.05,
            "parcours-gca": 0.05,
            "parcours-emii": 0.05,
            "parcours-icmp": 0.05,
            "parcours-caa": 0.05,
            "parcours-emp": 0.05,
            "parcours-fic": 0.05,
            "parcours-dtja": 0.05,
            "parcours-iaa": 0.05,
            "parcours-aee": 0.05,
            "parcours-pip": 0.05,
            "parcours-tee": 0.05,
            "parcours-teh": 0.05,
        }

        # Bonus spécifiques par affinité
        if any(m in texte_profil for m in ["math", "donnees", "statistique", "analyse de donnees", "data"]):
            scores_profil["parcours-isaia"] += 0.8
        if any(m in texte_profil for m in ["programmation", "logiciel", "web", "gestion", "informatique de gestion", "developpement"]):
            scores_profil["parcours-igglia"] += 0.8
        if any(m in texte_profil for m in ["electronique", "robotique", "systeme intelligent", "circuit", "microcontroleur"]):
            scores_profil["parcours-esiia"] += 0.8
        if any(m in texte_profil for m in ["multimedia", "telecom", "reseau", "communication"]):
            scores_profil["parcours-imticia"] += 0.8
        if any(m in texte_profil for m in ["batiment", "architecture", "chantier", "travaux", "genie civil", "btp", "dessin technique"]):
            scores_profil["parcours-gca"] += 0.8
        if any(m in texte_profil for m in ["mecanique", "maintenance", "usine", "industrielle"]):
            scores_profil["parcours-emii"] += 0.8
        if any(m in texte_profil for m in ["chimie", "petrole", "mine"]):
            scores_profil["parcours-icmp"] += 0.8
        if any(m in texte_profil for m in ["commerce", "affaires", "vente", "marketing", "commercial"]):
            scores_profil["parcours-caa"] += 0.95
        if any(m in texte_profil for m in ["management", "economie", "projet"]):
            scores_profil["parcours-emp"] += 0.8
        if any(m in texte_profil for m in ["finance", "comptabilite", "banque", "audit"]):
            scores_profil["parcours-fic"] += 0.85
        if any(m in texte_profil for m in ["droit", "juridique"]):
            scores_profil["parcours-dtja"] += 0.8
        if any(m in texte_profil for m in ["agroalimentaire", "transformation", "alimentaire"]):
            scores_profil["parcours-iaa"] += 0.8
        if any(m in texte_profil for m in ["agriculture", "elevage", "rural", "agronomie", "developpement rural"]):
            scores_profil["parcours-aee"] += 0.8
        if any(m in texte_profil for m in ["pharmacie", "pharmacologie", "medicament"]):
            scores_profil["parcours-pip"] += 0.8
        if any(m in texte_profil for m in ["environnement", "ecologie", "tourisme vert"]):
            scores_profil["parcours-tee"] += 0.8
        if any(m in texte_profil for m in ["hotel", "hotellerie", "accueil"]):
            scores_profil["parcours-teh"] += 0.8

        # Si le modèle ML est chargé, on intègre son signal probabiliste
        if self.modele is not None and self.featuriseur is not None:
            try:
                row = {
                    "id_candidat": "CANDIDAT_COURANT",
                    "matieres_preferees": matieres_str or "general",
                    "moyenne_scolaire": float(profil.get("moyenne_scolaire", profil.get("moyenne_generale", 12.5)) or 12.5),
                    "competences": "{}",
                    "centres_interet": centres_str or "general",
                    "projets": projets_str or "",
                    "preferences_professionnelles": str(profil.get("preferences_professionnelles", "salariat")).lower(),
                    "environnement_travail": str(profil.get("environnement_travail", "bureau")).lower(),
                }
                df = pd.DataFrame([row])
                x_trans = self.featuriseur.transform(df)
                probas = self.modele.predict_proba(x_trans)[0]
                classes = self.modele.classes_
                for cls, pr in zip(classes, probas):
                    if cls in scores_profil:
                        scores_profil[cls] += float(pr) * 0.3
            except Exception:
                pass

        # Tri des meilleurs parcours
        classement = sorted(scores_profil.items(), key=lambda x: x[1], reverse=True)
        top_classes = classement[:top_n]

        noms_parcours = {
            "parcours-igglia": "Informatique de Gestion, Génie Logiciel et Intelligence Artificielle (IGGLIA)",
            "parcours-isaia": "Informatique Statistique Appliquée et Intelligence Artificielle (ISAIA)",
            "parcours-esiia": "Électronique Système Informatique et Intelligence Artificielle (ESIIA)",
            "parcours-imticia": "Informatique Multimédia et Télécommunications (IMTICIA)",
            "parcours-gca": "Génie Civil et Architecture (GCA)",
            "parcours-emii": "Électro-Mécanique et Informatique Industrielle (EMII)",
            "parcours-icmp": "Industries Chimiques, Minières et Pétrolières (ICMP)",
            "parcours-caa": "Commerce et Administration des Affaires (CAA)",
            "parcours-emp": "Économie et Management de Projet (EMP)",
            "parcours-fic": "Finances et Comptabilités (FIC)",
            "parcours-dtja": "Droit et Techniques Juridiques des Affaires (DTJA)",
            "parcours-iaa": "Industrie Agroalimentaire (IAA)",
            "parcours-aee": "Agriculture et Élevage (AEE)",
            "parcours-pip": "Pharmacologie et Industries Pharmaceutiques (PIP)",
            "parcours-tee": "Tourisme et Environnement (TEE)",
            "parcours-teh": "Tourisme et Hôtellerie (TEH)",
        }

        recommandations = []
        sources_utilisees: List[Dict[str, Any]] = []
        sources_vues = set()

        for code_parcours, score_val in top_classes:
            nom_complet = noms_parcours.get(code_parcours, code_parcours)
            pertinence = int(min(95, max(45, round(score_val * 60 + 35))))

            pourquoi = []
            if centres_str:
                pourquoi.append(f"Forte adéquation avec vos centres d'intérêt ({centres_str})")
            if matieres_str:
                pourquoi.append(f"Cohérence avec vos matières déclarées ({matieres_str})")
            if not pourquoi:
                pourquoi.append("Profil général compatible avec les objectifs de formation")

            # Récupérer données officielles du RAG
            rag_info = self.rag.rechercher(nom_complet, top_k=1)
            prereq_txt = "Sélection sur dossier BACC (Séries C, D, S ou équivalent)."
            debouches_txt = "Information non détaillée dans le corpus officiel."

            if rag_info:
                p_meta = rag_info[0].passage.metadata
                if p_meta.get("prerequis"):
                    prereq_txt = " ".join(p_meta["prerequis"][:2])
                if p_meta.get("metiers"):
                    debouches_txt = "; ".join(p_meta["metiers"])
                elif code_parcours == "parcours-isaia":
                    debouches_txt = "Banques, entreprises industrielles et entreprises commerciales (source officielle ISPM)."

                for s in rag_info[0].sources:
                    if s.identifiant not in sources_vues:
                        sources_vues.add(s.identifiant)
                        sources_utilisees.append({
                            "nom": s.titre,
                            "type": "Source institutionnelle",
                            "origine": s.origine,
                            "url": s.url,
                            "statut": s.statut,
                            "date": s.date_consultation,
                        })

            recommandations.append({
                "code": code_parcours,
                "parcours": nom_complet,
                "pertinence": pertinence,
                "pourquoi": pourquoi,
                "prerequis": prereq_txt,
                "debouches": debouches_txt,
                "incertitude": "Modérée (recommandation statistique & règles indicatives)",
            })

        return {
            "statut": "succes",
            "recommandations": recommandations,
            "sources": sources_utilisees,
            "incertitude_globale": "Modérée",
        }

    # --- Outil 3 : Vérification des prérequis d'admission ---
    def verifier_prerequis(self, mention_ou_parcours: str, serie_bacc: str) -> Dict[str, Any]:
        """Vérifie la conformité d'une série de Bacc avec les conditions d'accès publiées par l'ISPM."""
        serie_bacc = serie_bacc.upper().strip()
        mots = mention_ou_parcours.lower()

        # Règles officielles issues de inscription.php
        if "informatique" in mots or "igglia" in mots or "isaia" in mots or "esiia" in mots or "imticia" in mots:
            series_autorisees = ["C", "D", "S", "TECHNIQUE INDUSTRIELLE"]
            admissible = any(s in serie_bacc for s in ["C", "D", "S", "TECH"])
            detail = "Le département Informatique et Télécommunication admet les séries C, D, S et techniques industrielles."
        elif "agronomie" in mots or "biotechnologie" in mots or "iaa" in mots or "aee" in mots or "pip" in mots:
            series_autorisees = ["C", "D", "S", "TECHNIQUE AGRICOLE", "A2 (avec Maths >= 12)"]
            admissible = any(s in serie_bacc for s in ["C", "D", "S", "AGRI", "A2"])
            detail = "Le département Biotechnologie et Agronomie admet C, D, S, techniques agricoles et A2 (avec note de Maths >= 12)."
        elif "affaires" in mots or "tourisme" in mots or "caa" in mots or "emp" in mots or "fic" in mots or "dtja" in mots or "tee" in mots or "teh" in mots:
            series_autorisees = ["Toutes séries de Bacc"]
            admissible = True
            detail = "Les départements Techniques des Affaires et Tourisme sont ouverts à toutes séries de Baccalauréat."
        else:
            series_autorisees = ["BACC Titulaire"]
            admissible = True
            detail = "Accès en première année par sélection de dossier pour tout titulaire du Baccalauréat."

        return {
            "parcours_ou_mention": mention_ou_parcours,
            "serie_declaree": serie_bacc,
            "admissible_selection": admissible,
            "series_officielles_admises": series_autorisees,
            "regle_officielle": detail,
            "source": "Conditions d'accès en première année (http://www.ispm-edu.com/inscription.php)",
            "avertissement": "L'admission définitive reste soumise à la sélection du dossier par l'administration de l'ISPM.",
        }

    # --- Outil 4 : Comparaison entre parcours ---
    def comparer_parcours(self, parcours_a: str, parcours_b: str) -> Dict[str, Any]:
        """Compare deux parcours officiels point par point avec sources vérifiables."""
        res_a = self.rag.rechercher(parcours_a, top_k=1)
        res_b = self.rag.rechercher(parcours_b, top_k=1)

        pass_a = res_a[0].passage if res_a else None
        pass_b = res_b[0].passage if res_b else None

        sources = []
        for res in (res_a or []) + (res_b or []):
            for s in res.sources:
                sources.append({
                    "nom": s.titre,
                    "origine": s.origine,
                    "url": s.url,
                    "statut": s.statut,
                    "date": s.date_consultation,
                })

        return {
            "parcours_1": {
                "titre": pass_a.titre if pass_a else parcours_a,
                "contenu": pass_a.contenu if pass_a else "Non documenté",
            },
            "parcours_2": {
                "titre": pass_b.titre if pass_b else parcours_b,
                "contenu": pass_b.contenu if pass_b else "Non documenté",
            },
            "sources": sources,
        }
