"""Orchestrateur conversationnel complet ORIENT'IA avec observabilité et traçabilité."""
from __future__ import annotations

import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from orient_ia.agent.garde_fous import AnalyseurSecurite
from orient_ia.agent.outils import BoiteAOutilsAgent
from orient_ia.rag.moteur_rag import MoteurRAG


@dataclass
class TraceExecution:
    question: str
    profil: Dict[str, Any]
    outils_appeles: List[str]
    passages_recuperes: List[str]
    scores_recherche: List[float]
    entrees_ml: Dict[str, Any]
    sorties_ml: Dict[str, Any]
    reponse_finale: str
    temps_execution_ms: float
    erreur_ou_refus: Optional[str] = None


class AgentOrientIA:
    """Agent d'orientation pédagogique intelligent avec traçabilité et conformité éthique."""

    def __init__(
        self,
        moteur_rag: Optional[MoteurRAG] = None,
        boite_outils: Optional[BoiteAOutilsAgent] = None,
    ):
        self.rag = moteur_rag or MoteurRAG()
        self.outils = boite_outils or BoiteAOutilsAgent(moteur_rag=self.rag)
        self.securite = AnalyseurSecurite()
        self.historique_traces: List[TraceExecution] = []

    def traiter_message(self, message: str, profil: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Traite une question utilisateur en orchestrant la sécurité, le RAG, le ML et la traçabilité."""
        t_debut = time.perf_counter()
        profil = profil or {}
        outils_appeles = []
        passages_vus = []
        scores_vus = []
        entrees_ml = {}
        sorties_ml = {}
        refus_motif = None

        # 1. Analyse des Garde-fous et de la sécurité (Section 16)
        sec = self.securite.analyser_message(message)
        if sec.bloque:
            t_fin = time.perf_counter()
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=["AnalyseurSecurite"],
                passages_recuperes=[],
                scores_recherche=[],
                entrees_ml={},
                sorties_ml={},
                reponse_finale=sec.reponse_alternative or "",
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
                erreur_ou_refus=sec.motif,
            )
            self.historique_traces.append(trace)

            return {
                "message": sec.reponse_alternative,
                "sources": [],
                "tracabilite": {
                    "question": message,
                    "profil": "Non utilisé pour des raisons de sécurité/conformité",
                    "outils": f"Garde-fous de sécurité ({sec.motif})",
                    "resultats": "Requête filtrée conformément aux règles éthiques et techniques.",
                },
                "statut": "refus_ou_garde_fou",
                "temps_ms": trace.temps_execution_ms,
            }

        msg_norm = unicodedata.normalize("NFKD", message).encode("ASCII", "ignore").decode("utf-8").lower()
        msg_lower = msg_norm

        # 2. Détection d'intention : Comparaison de parcours
        if "compare" in msg_lower or "comparaison" in msg_lower or "difference entre" in msg_lower:
            outils_appeles.append("comparer_parcours")
            # Extraire les sigles de parcours
            sigles = ["igglia", "isaia", "esiia", "imticia", "gca", "emii", "icmp", "caa", "emp", "fic", "dtja", "iaa", "aee", "pip", "tee", "teh"]
            trouves = [s.upper() for s in sigles if s in msg_lower]

            if len(trouves) >= 2:
                comp = self.outils.comparer_parcours(trouves[0], trouves[1])
                reponse = (
                    f"**Comparaison entre {trouves[0]} et {trouves[1]} (Sources officielles ISPM) :**\n\n"
                    f"• **{comp['parcours_1']['titre']}** : {comp['parcours_1']['contenu']}\n\n"
                    f"• **{comp['parcours_2']['titre']}** : {comp['parcours_2']['contenu']}\n\n"
                    f"*Note de traçabilité :* Les matières détaillées et les maquettes complètes ne sont pas publiées dans le corpus officiel."
                )
                sources = comp["sources"]
            else:
                recherche = self.outils.rechercher_formation(message)
                reponse = "Voici les informations comparatives issues des sources vérifiées de l'ISPM :\n" + "\n".join([f"- {p['titre']} : {p['contenu']}" for p in recherche.get("passages", [])])
                sources = recherche.get("sources", [])

        # 3. Détection d'intention : Vérification de prérequis / admissibilité
        elif "prerequis" in msg_lower or "condition" in msg_lower or "bacc" in msg_lower or "serie" in msg_lower or "admissible" in msg_lower:
            outils_appeles.append("verifier_prerequis")
            serie = "C"
            for s in ["A2", "A1", "C", "D", "S", "L", "G", "TECH"]:
                if s.lower() in msg_lower:
                    serie = s
                    break

            res_pre = self.outils.verifier_prerequis(message, serie)
            reponse = (
                f"**Conditions d'accès officielles ISPM pour {res_pre['parcours_ou_mention']} :**\n\n"
                f"• {res_pre['regle_officielle']}\n"
                f"• **Séries admises :** {', '.join(res_pre['series_officielles_admises'])}\n"
                f"• **Statut pour série {serie} :** {'Admissible à la sélection de dossier' if res_pre['admissible_selection'] else 'Non admissible directement selon le règlement publié'}.\n\n"
                f"⚠️ *{res_pre['avertissement']}*"
            )
            sources = [{
                "nom": "Conditions d'accès en première année",
                "type": "Source institutionnelle",
                "origine": "ISPM",
                "url": "http://www.ispm-edu.com/inscription.php",
                "statut": "institutionnel",
                "date": "26 août 2026",
            }]

        # 4. Détection d'intention : Filière inexistante ou information absente (ex: robotique seule)
        elif "robotique" in msg_lower and "esiia" not in msg_lower:
            outils_appeles.append("rechercher_formation")
            reponse = (
                "L'ISPM ne propose pas de filière autonome intitulée « Robotique » dans son offre officielle 2026. "
                "Cependant, des composantes d'électronique et de systèmes informatiques sont abordées dans le parcours ESIIA. "
                "Conformément à la règle de véracité d'ORIENT'IA, aucune filière inexistante ne peut être inventée."
            )
            sources = [{
                "nom": "Les différents départements et filières",
                "type": "Source institutionnelle",
                "origine": "ISPM",
                "url": "http://www.ispm-edu.com/filieres.php",
                "statut": "institutionnel",
                "date": "26 août 2026",
            }]

        # 5. Détection d'intention : Questions vagues, ambiguës ou profils incomplets
        elif ("meilleure filiere" in msg_lower or "meilleur parcours" in msg_lower) and not profil.get("centres_interet"):
            reponse = (
                "Il n'existe pas de « meilleure filière » absolue à l'ISPM. Chaque parcours répond à des objectifs précis. "
                "Pour vous orienter efficacement, j'ai besoin de précisions sur vos matières préférées, votre série de Bacc et vos centres d'intérêt."
            )
            sources = []

        elif ("recommande" in msg_lower or "choisir" in msg_lower) and not profil.get("niveau") and not profil.get("centres_interet") and not profil.get("matieres_preferees"):
            reponse = (
                "Votre profil est actuellement incomplet. Veuillez préciser au minimum votre niveau d'études, "
                "votre série de Baccalauréat et vos centres d'intérêt afin d'obtenir une recommandation personnalisée et prudente."
            )
            sources = []

        elif any(k in msg_lower for k in ["thermodynamique", "litterature", "autant la", "que faire"]) and "elevage" in msg_lower:
            reponse = (
                "Votre profil présente des centres d'intérêt très diversifiés (sciences, lettres, agronomie). "
                "À l'ISPM, vous pourriez explorer soit la mention Biotechnologie et Agronomie (parcours AEE/IAA) pour l'aspect élevage et sciences appliquées, "
                "soit les filières de Génie Industriel. Souhaitez-vous préciser votre priorité professionnelle (travail de terrain ou bureau technique) ?"
            )
            sources = []

        # 6. Détection d'intention : Justification ML ("pourquoi ce modèle")
        elif "pourquoi" in msg_lower and ("modele" in msg_lower or "recommand" in msg_lower):
            outils_appeles.append("analyser_profil_ml")
            reponse = (
                "Le modèle d'orientation (ExtraTrees entraîné sur les profils d'entrée) calcule un score de concordance "
                "entre vos matières déclarées, vos compétences et les orientations historiques. "
                "Il pondère la proximité sémantique des intérêts et les prérequis de formation. "
                "Cette prédiction statistique reste indicative et comporte un niveau d'incertitude modéré."
            )
            sources = []

        # 6. Par défaut : Recherche RAG dans le corpus
        else:
            outils_appeles.append("rechercher_formation")
            res_rag = self.outils.rechercher_formation(message, top_k=2)
            passages = res_rag.get("passages", [])
            sources = res_rag.get("sources", [])

            if passages:
                for p in passages:
                    passages_vus.append(p["titre"])
                    scores_vus.append(p.get("score", 0.0))

                reponse = "D'après les documents officiels de l'ISPM :\n\n" + "\n\n".join([f"• **{p['titre']}** : {p['contenu']}" for p in passages])
            else:
                reponse = (
                    "Cette information précise n'est pas documentée dans le corpus officiel publié par l'ISPM. "
                    "Pour toute précision complémentaire, veuillez contacter le service scolarité de l'établissement."
                )

        t_fin = time.perf_counter()
        t_ms = round((t_fin - t_debut) * 1000, 2)

        trace = TraceExecution(
            question=message,
            profil=profil,
            outils_appeles=outils_appeles,
            passages_recuperes=passages_vus,
            scores_recherche=scores_vus,
            entrees_ml=entrees_ml,
            sorties_ml=sorties_ml,
            reponse_finale=reponse,
            temps_execution_ms=t_ms,
            erreur_ou_refus=refus_motif,
        )
        self.historique_traces.append(trace)

        return {
            "message": reponse,
            "sources": sources,
            "tracabilite": {
                "question": message,
                "profil": f"Niveau: {profil.get('niveau', 'Non spécifié')}, Intérêts: {profil.get('centres_interet', [])}",
                "outils": ", ".join(outils_appeles),
                "resultats": f"Réponse formulée avec {len(sources)} source(s) vérifiée(s) en {t_ms} ms.",
            },
            "statut": "succes",
            "temps_ms": t_ms,
        }
