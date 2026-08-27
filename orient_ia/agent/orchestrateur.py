"""Orchestrateur conversationnel complet ORIENT'IA avec observabilité et traçabilité."""
from __future__ import annotations

import time
import unicodedata
from collections.abc import Mapping
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

        # 1. Analyse des Garde-fous et de la sécurité
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
                "statut": "refus_ou_garde_fou",
                "etat": "refus_ethique" if "psychologique" in str(sec.motif) or "genre" in str(sec.motif) else "bloque_securite",
                "message": sec.reponse_alternative,
                "reponse": sec.reponse_alternative,
                "sources": [],
                "trace": asdict(trace),
            }

        # 2. Détection d'intentions
        msg_norm = unicodedata.normalize('NFKD', message).encode('ASCII', 'ignore').decode('utf-8').lower()

        # A. Questions sur des informations non publiées / absentes du corpus
        if any(w in msg_norm for w in ["volume horaire", "combien d'heures", "tarif", "frais de scolarite", "prix", "cout", "programme detaille", "par semestre", "passerelle", "bourse", "logement", "dortoir"]):
            reponse = (
                "Cette information spécifique (volume horaire précis, tarifs des frais de scolarité, passerelles officielles non déclarées ou syllabus semestriel) "
                "n'est pas présente dans les sources officielles de l'ISPM actuellement référencées."
            )
            t_fin = time.perf_counter()
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=["reconnaissance_absence"],
                passages_recuperes=[],
                scores_recherche=[],
                entrees_ml={},
                sorties_ml={},
                reponse_finale=reponse,
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
            )
            self.historique_traces.append(trace)
            return {
                "statut": "information_non_disponible",
                "etat": "information_non_disponible",
                "message": reponse,
                "reponse": reponse,
                "sources": [],
                "trace": asdict(trace),
            }

        # B. Vérification de prérequis / Bacc
        if any(w in msg_norm for w in ["bacc", "serie", "admissib", "prerequis", "condition d'acces", "inscription"]):
            outils_appeles.append("verifier_prerequis")
            # Extraction sommaire de la série
            serie = "C" if "serie c" in msg_norm or "bacc c" in msg_norm else ("D" if "serie d" in msg_norm or "bacc d" in msg_norm else ("A2" if "serie a" in msg_norm or "bacc a" in msg_norm else "Non précisée"))
            verif = self.outils.verifier_prerequis(message, serie)
            t_fin = time.perf_counter()
            reponse = f"{verif['regle_officielle']}\n\n*Source vérifiée : {verif['source']}*.\n{verif['avertissement']}"
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=outils_appeles,
                passages_recuperes=[verif['regle_officielle']],
                scores_recherche=[1.0],
                entrees_ml={"serie": serie, "demande": message},
                sorties_ml=verif,
                reponse_finale=reponse,
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
            )
            self.historique_traces.append(trace)
            return {
                "statut": "succes",
                "etat": "succes",
                "message": reponse,
                "reponse": reponse,
                "sources": [{"titre": "Conditions d'inscription ISPM", "url": "http://www.ispm-edu.com/inscription.php", "statut": "institutionnel"}],
                "trace": asdict(trace),
            }

        # B. Comparaison explicite entre parcours
        if any(w in msg_norm for w in ["difference entre", "comparer", "vs", "versus", "ou choisir entre"]):
            outils_appeles.append("comparer_parcours")
            p1 = "igglia" if "igglia" in msg_norm else ("isaia" if "isaia" in msg_norm else "Informatique")
            p2 = "esiia" if "esiia" in msg_norm else ("imticia" if "imticia" in msg_norm else "Électronique")
            comp = self.outils.comparer_parcours(p1, p2)
            t_fin = time.perf_counter()
            reponse = (
                f"### Comparaison institutionnelle :\n"
                f"- **{comp['parcours_1']['titre']}** : {comp['parcours_1']['contenu']}\n\n"
                f"- **{comp['parcours_2']['titre']}** : {comp['parcours_2']['contenu']}\n\n"
                f"*Toutes les données proviennent des présentations officielles des filières ISPM.*"
            )
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=outils_appeles,
                passages_recuperes=[comp['parcours_1']['contenu'], comp['parcours_2']['contenu']],
                scores_recherche=[1.0, 1.0],
                entrees_ml={"p1": p1, "p2": p2},
                sorties_ml=comp,
                reponse_finale=reponse,
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
            )
            self.historique_traces.append(trace)
            return {
                "statut": "succes",
                "etat": "succes",
                "message": reponse,
                "reponse": reponse,
                "sources": comp["sources"],
                "trace": asdict(trace),
            }

        # C. Question ambiguë / qualitative générale
        if any(w in msg_norm for w in ["meilleure filiere", "meilleur parcours", "meilleure formation", "classement", "laquelle est la meilleure"]):
            reponse = "Il n'y a pas de 'meilleure filière' absolue à l'ISPM. Le choix optimal dépend entièrement de votre profil, de votre série de Baccalauréat et de vos objectifs. Veuillez préciser vos centres d'intérêt pour que je puisse vous guider."
            t_fin = time.perf_counter()
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=["gestion_ambiguite"],
                passages_recuperes=[],
                scores_recherche=[],
                entrees_ml={},
                sorties_ml={},
                reponse_finale=reponse,
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
            )
            self.historique_traces.append(trace)
            return {
                "statut": "demande_precision",
                "etat": "demande_precision",
                "message": reponse,
                "reponse": reponse,
                "sources": [],
                "trace": asdict(trace),
            }

        # D. Recommandation / Orientation via Profil ML
        if any(w in msg_norm for w in ["recommand", "orienter", "quel parcours", "que faire", "conseil"]):
            outils_appeles.append("analyser_profil_ml")
            rec_res = self.outils.analyser_profil_ml(profil)
            entrees_ml = profil
            sorties_ml = rec_res
            recs = rec_res.get("recommandations", [])
            if recs and profil.get("niveau"):
                lignes = []
                for r in recs:
                    lignes.append(f"- **{r['parcours']}** (Indice de pertinence : {r['pertinence']}%) : {r['pourquoi'][0] if r['pourquoi'] else ''}")
                reponse = "Voici les pistes recommandées à partir de vos centres d'intérêt et matières déclarées :\n\n" + "\n".join(lignes)
                reponse += "\n\n*Note : Ces recommandations statistiques sont données à titre indicatif et ne remplacent pas les conditions d'admission officielles.*"
            else:
                reponse = "Votre profil est incomplet (niveau d'étude manquant). Veuillez préciser vos matières préférées ou centres d'intérêt pour orienter la recommandation."

            t_fin = time.perf_counter()
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=outils_appeles,
                passages_recuperes=[r['parcours'] for r in recs],
                scores_recherche=[r['pertinence'] / 100.0 for r in recs],
                entrees_ml=entrees_ml,
                sorties_ml=sorties_ml,
                reponse_finale=reponse,
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
            )
            self.historique_traces.append(trace)
            return {
                "statut": "succes",
                "etat": "succes",
                "message": reponse,
                "reponse": reponse,
                "sources": rec_res.get("sources", []),
                "trace": asdict(trace),
            }

        # E. Recherche documentaire générale RAG
        outils_appeles.append("rechercher_formation")
        recherche = self.outils.rechercher_formation(message, top_k=2)
        passages = recherche.get("passages", [])
        sources = recherche.get("sources", [])

        if passages and passages[0]["score"] >= 0.28:
            contenus = [f"**{p['titre']}** : {p['contenu']}" for p in passages]
            reponse = "D'après les documents institutionnels de l'ISPM :\n\n" + "\n\n".join(contenus)
            etat = "succes"
        else:
            reponse = (
                "Cette information ou filière spécifique n'est pas présente dans les mentions et parcours officiels de l'ISPM actuellement indexés."
            )
            etat = "information_non_disponible"

        t_fin = time.perf_counter()
        trace = TraceExecution(
            question=message,
            profil=profil,
            outils_appeles=outils_appeles,
            passages_recuperes=[p["contenu"] for p in passages],
            scores_recherche=[p["score"] for p in passages],
            entrees_ml={},
            sorties_ml=recherche,
            reponse_finale=reponse,
            temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
        )
        self.historique_traces.append(trace)

        return {
            "statut": "succes" if etat == "succes" else "information_non_disponible",
            "etat": etat,
            "message": reponse,
            "reponse": reponse,
            "sources": sources,
            "trace": asdict(trace),
        }


# Instance globale partagée
_agent_partage = AgentOrientIA()


def traiter_message(
    message: str,
    profil: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Point d'entrée standardisé pour l'interaction avec l'Agent conversationnel."""
    profil_dict = dict(profil) if profil else {}
    return _agent_partage.traiter_message(message, profil_dict)


def orchestrer_conversation(
    message: str,
    profil: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Alias fonctionnel pour l'orchestration conversationnelle."""
    return traiter_message(message=message, profil=profil, session_id=session_id)
