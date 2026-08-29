"""Boîte à outils fonctionnels pour l'Agent ORIENT'IA."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from orient_ia.agent.parcours import extraire_codes_parcours, fiche_depuis_passage
from orient_ia.rag.moteur_rag import MoteurRAG, SourceReference
from orient_ia.rag.service import rechercher_documents


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

    # --- Outil 2 : Inférence du profil candidat avec modèle ML ---
    def analyser_profil_ml(self, profil_candidat: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
        """Analyse le profil du candidat via le modèle de Machine Learning et le moteur RAG."""
        import unicodedata

        def _clean_str(txt: Any) -> str:
            raw = ", ".join(txt) if isinstance(txt, list) else str(txt or "")
            return unicodedata.normalize('NFKD', raw).encode('ASCII', 'ignore').decode('utf-8').lower()

        centres_raw = profil_candidat.get("centres_interet", [])
        centres_str = ", ".join(centres_raw) if isinstance(centres_raw, list) else str(centres_raw or "")
        centres_clean = _clean_str(centres_raw)

        matieres_raw = profil_candidat.get("matieres_preferees", [])
        matieres_str = ", ".join(matieres_raw) if isinstance(matieres_raw, list) else str(matieres_raw or "")
        matieres_clean = _clean_str(matieres_raw)

        projets_raw = profil_candidat.get("projets", [])
        projets_clean = _clean_str(projets_raw)

        texte_global = f"{centres_clean} {matieres_clean} {projets_clean}".strip()

        # Initialisation neutre de tous les 16 parcours
        scores_profil = {
            "parcours-igglia": 0.1,
            "parcours-isaia": 0.1,
            "parcours-esiia": 0.1,
            "parcours-imticia": 0.1,
            "parcours-emii": 0.1,
            "parcours-icmp": 0.1,
            "parcours-gca": 0.1,
            "parcours-caa": 0.1,
            "parcours-emp": 0.1,
            "parcours-fic": 0.1,
            "parcours-dtja": 0.1,
            "parcours-iaa": 0.1,
            "parcours-aee": 0.1,
            "parcours-pip": 0.1,
            "parcours-tee": 0.1,
            "parcours-teh": 0.1,
        }

        raisons_parcours: Dict[str, List[str]] = {k: [] for k in scores_profil}

        # 1. Santé, Pharmacologie et Biotechnologie
        if any(w in texte_global for w in ["sante", "pharma", "medicament", "galenique", "medical"]):
            scores_profil["parcours-pip"] += 1.4
            scores_profil["parcours-iaa"] += 0.6
            raisons_parcours["parcours-pip"].append("Adéquation directe avec votre intérêt pour la santé et l'industrie pharmaceutique")
            raisons_parcours["parcours-iaa"].append("Synergie entre sciences de la santé et biochimie alimentaire")
        if any(w in texte_global for w in ["biologie", "agroalimentaire", "alimentaire", "haccp", "nutrition"]):
            scores_profil["parcours-iaa"] += 1.3
            raisons_parcours["parcours-iaa"].append("Cohérence avec les sciences biologiques et la transformation alimentaire")

        # 2. Écologie, Environnement et Agronomie
        if any(w in texte_global for w in ["ecologie", "environnement", "biodiversite", "parc", "nature"]):
            scores_profil["parcours-tee"] += 1.4
            scores_profil["parcours-aee"] += 0.6
            raisons_parcours["parcours-tee"].append("Forte adéquation avec votre intérêt pour l'écologie et l'environnement durable")
            raisons_parcours["parcours-aee"].append("Lien direct entre préservation écologique et agronomie durable")
        if any(w in texte_global for w in ["agri", "agriculture", "elevage", "rural", "sol", "zootechnie"]):
            scores_profil["parcours-aee"] += 1.4
            raisons_parcours["parcours-aee"].append("Correspondance directe avec la gestion des productions agricoles et l'élevage")

        # 3. Informatique, Logiciel, Données et IA
        if any(w in texte_global for w in ["donnee", "data", "stat", "econometrie", "predicti"]):
            scores_profil["parcours-isaia"] += 1.5
            raisons_parcours["parcours-isaia"].append("Excellente synergie avec l'analyse de données statistiques et le machine learning")
        if any(w in texte_global for w in ["logiciel", "developpement", "web", "programmation", "algo", "application"]):
            scores_profil["parcours-igglia"] += 1.3
            raisons_parcours["parcours-igglia"].append("Alignement avec vos aptitudes en génie logiciel et développement d'applications")
        if any(w in texte_global for w in ["telecom", "multimedia", "reseau", "cloud"]):
            scores_profil["parcours-imticia"] += 1.3
            raisons_parcours["parcours-imticia"].append("Adéquation avec les technologies web, cloud et télécommunications")
        if any(w in texte_global for w in ["info", "intelligence artificielle", "ia"]) and not any(w in texte_global for w in ["donnee", "data", "stat"]):
            scores_profil["parcours-igglia"] += 0.8
            scores_profil["parcours-imticia"] += 0.5
            scores_profil["parcours-esiia"] += 0.4
            scores_profil["parcours-isaia"] += 0.4

        # 4. Électronique, Robotique, Automatisme et Mécanique
        if any(w in texte_global for w in ["electro", "embarque", "robot", "circuit", "microcontroleur", "iot", "materiel"]):
            scores_profil["parcours-esiia"] += 1.4
            raisons_parcours["parcours-esiia"].append("Alignement direct avec les systèmes électroniques et l'informatique embarquée")
        if any(w in texte_global for w in ["mecanique", "automatisme", "maintenance", "usine", "gmao"]):
            scores_profil["parcours-emii"] += 1.4
            raisons_parcours["parcours-emii"].append("Correspondance avec l'électromécanique et la maintenance industrielle")

        # 5. BTP, Génie Civil et Architecture
        if any(w in texte_global for w in ["batiment", "architecture", "genie civil", "construction", "chantier", "ouvrage", "btp", "dessin technique", "plans", "structure", "design"]):
            scores_profil["parcours-gca"] += 1.4
            raisons_parcours["parcours-gca"].append("Forte cohérence avec les métiers du bâtiment, travaux publics et architecture")

        # 6. Chimie, Mines et Pétrole
        if any(w in texte_global for w in ["chimie", "mine", "petrole", "raffinage", "qhse"]):
            scores_profil["parcours-icmp"] += 1.4
            raisons_parcours["parcours-icmp"].append("Spécialisation adaptée aux procédés chimiques, miniers et pétroliers")

        # 7. Management, Affaires, Commerce et Finance
        if any(w in texte_global for w in ["commerce", "vente", "marketing", "negociation", "client"]):
            scores_profil["parcours-caa"] += 1.4
            raisons_parcours["parcours-caa"].append("Orientation ciblée vers la stratégie commerciale et le développement des ventes")
        if any(w in texte_global for w in ["manage", "management", "projet", "organisation", "planification", "strategie"]):
            scores_profil["parcours-emp"] += 1.3
            raisons_parcours["parcours-emp"].append("Correspondance avec le management stratégique et la conduite de projet")
        if any(w in texte_global for w in ["compta", "comptabilite", "audit", "fiscalite", "tresorerie", "finance", "banque"]):
            scores_profil["parcours-fic"] += 1.4
            raisons_parcours["parcours-fic"].append("Adéquation avec la gestion comptable, l'audit et la finance d'entreprise")
        if any(w in texte_global for w in ["droit", "juridique", "contrat", "contentieux", "loi"]):
            scores_profil["parcours-dtja"] += 1.4
            raisons_parcours["parcours-dtja"].append("Forte affinité avec le droit des affaires et le conseil juridique")

        # 8. Tourisme et Hôtellerie
        if any(w in texte_global for w in ["tourisme", "hotel", "hotellerie", "hebergement", "restauration", "voyage", "accueil"]):
            scores_profil["parcours-teh"] += 1.4
            scores_profil["parcours-tee"] += 0.8
            raisons_parcours["parcours-teh"].append("Adéquation avec le management hôtelier et les services d'hébergement")

        # Inférence ML avec le modèle ExtraTrees si des données textuelles significatives sont présentes
        if self.modele and self.featuriseur and texte_global:
            try:
                row_dict = {
                    "matieres_preferees": matieres_clean or "general",
                    "moyenne_scolaire": float(profil_candidat.get("moyenne_scolaire", 14.0)),
                    "centres_interet": centres_clean or "polytechnique",
                    "projets": projets_clean or "projet_academique",
                    "preferences_professionnelles": str(profil_candidat.get("preferences_professionnelles", "salariat")),
                    "environnement_travail": str(profil_candidat.get("environnement_travail", "hybride")),
                }
                df_candidat = pd.DataFrame([row_dict])
                X_cand = self.featuriseur.transform(df_candidat)
                probas = self.modele.predict_proba(X_cand)[0]
                classes = self.modele.classes_

                for cls, pr in zip(classes, probas):
                    if cls in scores_profil:
                        scores_profil[cls] += float(pr) * 0.4
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
            pertinence = int(min(95, max(45, round(score_val * 40 + 40))))

            pourquoi = raisons_parcours.get(code_parcours, [])
            if not pourquoi:
                if centres_str:
                    pourquoi.append(f"Compatibilité avec vos centres d'intérêt généraux ({centres_str})")
                elif matieres_str:
                    pourquoi.append(f"Cohérence avec vos matières déclarées ({matieres_str})")
                else:
                    pourquoi.append("Profil général compatible avec les objectifs de la formation")

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

    def resoudre_fiche_parcours(self, requete: str) -> Optional[Dict[str, Any]]:
        """Résout une requête vers une fiche de parcours officielle (alias, sigle ou recherche)."""
        if not requete or not str(requete).strip():
            return None

        codes = extraire_codes_parcours(requete)
        if codes:
            passage = self.rag.obtenir_par_identifiant(codes[0])
            if passage:
                return fiche_depuis_passage(passage)

        passage = self.rag.obtenir_par_identifiant(str(requete).strip())
        if passage and passage.categorie == "parcours":
            return fiche_depuis_passage(passage)

        resultats = self.rag.rechercher(requete, top_k=6)
        for res in resultats:
            if res.passage.categorie == "parcours":
                return fiche_depuis_passage(res.passage)
        return None

    def _sources_depuis_fiches(self, *codes: str) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        vues = set()
        for code in codes:
            if not code:
                continue
            for s in self.rag.get_sources_parcours(code):
                if s.identifiant in vues:
                    continue
                vues.add(s.identifiant)
                sources.append({
                    "nom": s.titre,
                    "origine": s.origine,
                    "url": s.url,
                    "statut": s.statut,
                    "date": s.date_consultation,
                })
        return sources

    # --- Outil 4 : Comparaison entre parcours ---
    def comparer_parcours(self, parcours_a: str, parcours_b: str) -> Dict[str, Any]:
        """Compare deux parcours officiels point par point avec sources vérifiables."""
        fiche_a = self.resoudre_fiche_parcours(parcours_a)
        fiche_b = self.resoudre_fiche_parcours(parcours_b)

        sources = self._sources_depuis_fiches(
            (fiche_a or {}).get("code", ""),
            (fiche_b or {}).get("code", ""),
        )

        def _bloc(fiche: Optional[Dict[str, Any]], brut: str) -> Dict[str, Any]:
            if not fiche:
                return {"titre": brut, "contenu": "Non documenté", "fiche": None}
            return {
                "titre": fiche["nom"],
                "contenu": fiche["contenu"],
                "fiche": fiche,
            }

        return {
            "parcours_1": _bloc(fiche_a, parcours_a),
            "parcours_2": _bloc(fiche_b, parcours_b),
            "sources": sources,
            "statut": "ok" if fiche_a and fiche_b else ("not_found" if not fiche_a and not fiche_b else "incomplet"),
        }


# --- Fonctions compatibles standalone ---
_boite_defaut = BoiteAOutilsAgent()


def rechercher_formations(question: str, nombre_resultats: int = 5, seuil: float = 0.05) -> dict[str, Any]:
    """Recherche documentaire locale dans le corpus pédagogique."""
    return rechercher_documents(question=question, nombre_resultats=nombre_resultats, seuil=seuil)


def analyser_profil(profil: Mapping[str, Any] | None) -> dict[str, Any]:
    """Exécute l'inférence du moteur ML à partir d'un profil candidat compatible."""
    if not profil:
        return {"statut": "incomplet", "message": "Profil insuffisant pour une recommandation fiable."}
    return _boite_defaut.analyser_profil_ml(dict(profil))


def comparer_parcours(parcours: Iterable[str] | str | None, parcours_b: str | None = None) -> dict[str, Any]:
    """Compare plusieurs parcours de formation."""
    if isinstance(parcours, str) and parcours_b:
        return _boite_defaut.comparer_parcours(parcours, parcours_b)
    if parcours is None:
        return {"statut": "incomplet", "message": "Aucun parcours fourni pour la comparaison."}
    if isinstance(parcours, str):
        liste = [p.strip() for p in parcours.split(",") if p.strip()]
    else:
        liste = [str(item).strip() for item in parcours if str(item).strip()]
    if len(liste) >= 2:
        return _boite_defaut.comparer_parcours(liste[0], liste[1])
    elif len(liste) == 1:
        return _boite_defaut.comparer_parcours(liste[0], "")
    return {"statut": "incomplet", "message": "Aucun parcours fourni pour la comparaison."}
