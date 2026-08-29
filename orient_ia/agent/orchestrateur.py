"""Orchestrateur conversationnel complet ORIENT'IA avec observabilité et traçabilité."""
from __future__ import annotations

import time
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from orient_ia.agent.garde_fous import AnalyseurSecurite
from orient_ia.agent.outils import BoiteAOutilsAgent
from orient_ia.agent.parcours import (
    PAIRES_PROCHES,
    detecter_intention,
    extraire_codes_parcours,
    extraire_serie_bacc,
    formater_comparaison,
    formater_fiche,
    formater_series,
)
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

        # 2. Détection d'intentions et des filières citées
        msg_norm = unicodedata.normalize("NFKD", message).encode("ASCII", "ignore").decode("utf-8").lower()
        intention = detecter_intention(message)
        codes = extraire_codes_parcours(message)
        serie = extraire_serie_bacc(message)

        def _repondre(
            reponse: str,
            etat: str,
            sources: Optional[List[Dict[str, Any]]] = None,
            sorties: Optional[Dict[str, Any]] = None,
            passages: Optional[List[str]] = None,
            scores: Optional[List[float]] = None,
        ) -> Dict[str, Any]:
            t_fin = time.perf_counter()
            trace = TraceExecution(
                question=message,
                profil=profil,
                outils_appeles=outils_appeles,
                passages_recuperes=passages or [],
                scores_recherche=scores or [],
                entrees_ml=entrees_ml,
                sorties_ml=sorties or {"intention": intention, "parcours": codes},
                reponse_finale=reponse,
                temps_execution_ms=round((t_fin - t_debut) * 1000, 2),
            )
            self.historique_traces.append(trace)
            statut = "succes" if etat == "succes" else etat
            return {
                "statut": statut,
                "etat": etat,
                "message": reponse,
                "reponse": reponse,
                "sources": sources or [],
                "trace": asdict(trace),
            }

        # A. Informations non publiées
        if any(
            w in msg_norm
            for w in [
                "volume horaire",
                "combien d'heures",
                "tarif",
                "frais de scolarite",
                "prix",
                "cout",
                "programme detaille",
                "par semestre",
                "passerelle",
                "bourse",
                "logement",
                "dortoir",
            ]
        ):
            outils_appeles.append("reconnaissance_absence")
            return _repondre(
                (
                    "Cette information précise (volume horaire, tarifs, passerelles ou syllabus semestriel) "
                    "n'est pas publiée dans les sources officielles ISPM actuellement indexées."
                ),
                "information_non_disponible",
            )

        # B. Comparaison réelle des filières citées (plus de repli Informatique/Électronique)
        if intention == "comparaison":
            outils_appeles.append("comparer_parcours")
            c1 = codes[0] if len(codes) >= 1 else None
            c2 = codes[1] if len(codes) >= 2 else PAIRES_PROCHES.get(c1 or "")
            if not c1:
                return _repondre(
                    "Pour comparer, nommez deux filières ISPM (ex. TEE et TEH, IGGLIA et ISAIA, GCA et EMII).",
                    "demande_precision",
                )
            if not c2:
                return _repondre(
                    f"J'ai identifié **{c1.replace('parcours-', '').upper()}**. "
                    "Indiquez la seconde filière à comparer.",
                    "demande_precision",
                )
            comp = self.outils.comparer_parcours(c1, c2)
            f1 = comp["parcours_1"].get("fiche")
            f2 = comp["parcours_2"].get("fiche")
            if not f1 or not f2:
                return _repondre(
                    "Je n'ai pas trouvé les deux filières dans le corpus officiel ISPM. "
                    "Vérifiez les sigles (TEE, TEH, IGGLIA, GCA, etc.).",
                    "information_non_disponible",
                    sources=comp.get("sources", []),
                    sorties=comp,
                )
            return _repondre(
                formater_comparaison(f1, f2),
                "succes",
                sources=comp.get("sources", []),
                sorties=comp,
                passages=[f1.get("contenu", ""), f2.get("contenu", "")],
                scores=[1.0, 1.0],
            )

        # C. Prérequis / série de bac — réponse ciblée, pas un texte générique Informatique
        if intention == "prerequis":
            outils_appeles.append("verifier_prerequis")
            sources_prereq = [{
                "titre": "Conditions d'inscription ISPM",
                "url": "http://www.ispm-edu.com/inscription.php",
                "statut": "institutionnel",
            }]
            if codes:
                fiche = self.outils.resoudre_fiche_parcours(codes[0])
                if fiche:
                    reponse = formater_fiche(fiche, "prerequis")
                    if serie:
                        verif = self.outils.verifier_prerequis(fiche["nom"], serie)
                        reponse += (
                            f"\n\nPour un **Bac {serie}** : {verif['regle_officielle']} "
                            f"{verif['avertissement']}"
                        )
                    return _repondre(
                        reponse,
                        "succes",
                        sources=sources_prereq,
                        sorties=fiche,
                        passages=fiche.get("prerequis", []),
                    )
            if serie:
                return _repondre(
                    formater_series(serie),
                    "succes",
                    sources=sources_prereq,
                    sorties={"serie": serie},
                )
            verif = self.outils.verifier_prerequis(message, "Non précisée")
            return _repondre(
                f"{verif['regle_officielle']}\n\n*{verif['avertissement']}*",
                "succes",
                sources=sources_prereq,
                sorties=verif,
            )

        # D. Ambiguïté qualitative
        if intention == "ambiguite":
            outils_appeles.append("gestion_ambiguite")
            return _repondre(
                (
                    "Il n'existe pas de « meilleure filière » à l'ISPM. "
                    "Le bon choix dépend de vos matières, de votre série de bac et du métier visé. "
                    "Cochez vos centres d'intérêt à gauche, ou comparez deux sigles (ex. TEE et TEH)."
                ),
                "demande_precision",
            )

        # E. Recommandation de profil
        if intention == "recommandation":
            outils_appeles.append("analyser_profil_ml")
            rec_res = self.outils.analyser_profil_ml(profil)
            entrees_ml = profil
            recs = rec_res.get("recommandations", [])
            if recs and (profil.get("niveau") or profil.get("centres_interet") or profil.get("matieres_preferees")):
                lignes = [
                    f"- **{r['parcours']}** ({r['pertinence']}%) : {r['pourquoi'][0] if r['pourquoi'] else ''}"
                    for r in recs
                ]
                reponse = (
                    "Pistes cohérentes avec le profil déclaré :\n\n"
                    + "\n".join(lignes)
                    + "\n\n*Repères indicatifs, distincts de la sélection officielle des dossiers.*"
                )
            else:
                reponse = (
                    "Pour une recommandation utile, indiquez au moins un centre d'intérêt, "
                    "des matières fortes, ou lancez « Analyser mon profil »."
                )
            return _repondre(
                reponse,
                "succes",
                sources=rec_res.get("sources", []),
                sorties=rec_res,
                passages=[r["parcours"] for r in recs],
                scores=[r["pertinence"] / 100.0 for r in recs],
            )

        # F. Question ciblée sur une filière identifiée
        if codes:
            outils_appeles.append("rechercher_formation")
            fiche = self.outils.resoudre_fiche_parcours(codes[0])
            if fiche:
                intention_fiche = intention if intention in {"metiers", "competences", "matieres", "prerequis"} else "fiche"
                return _repondre(
                    formater_fiche(fiche, intention_fiche),
                    "succes",
                    sources=self.outils._sources_depuis_fiches(fiche["code"]),
                    sorties=fiche,
                    passages=[fiche.get("contenu", "")],
                    scores=[1.0],
                )

        # G. Recherche documentaire générale, sans dump de fiches hétérogènes
        outils_appeles.append("rechercher_formation")
        recherche = self.outils.rechercher_formation(message, top_k=4)
        passages = recherche.get("passages", [])
        sources = recherche.get("sources", [])
        parcours_hits = [p for p in passages if p.get("categorie") == "parcours"]

        if parcours_hits and parcours_hits[0]["score"] >= 0.28:
            fiche = self.outils.resoudre_fiche_parcours(parcours_hits[0].get("titre", message))
            if fiche:
                intention_fiche = intention if intention in {"metiers", "competences", "matieres"} else "fiche"
                reponse = formater_fiche(fiche, intention_fiche)
            else:
                p = parcours_hits[0]
                reponse = f"**{p['titre']}**\n\n{p['contenu']}"
            return _repondre(
                reponse,
                "succes",
                sources=sources,
                sorties=recherche,
                passages=[p["contenu"] for p in parcours_hits[:2]],
                scores=[p["score"] for p in parcours_hits[:2]],
            )

        if passages and passages[0]["score"] >= 0.35:
            p = passages[0]
            return _repondre(
                f"{p['titre']}\n\n{p['contenu']}",
                "succes",
                sources=sources,
                sorties=recherche,
                passages=[p["contenu"]],
                scores=[p["score"]],
            )

        return _repondre(
            (
                "Je n'ai pas trouvé cette information dans les mentions et parcours officiels de l'ISPM. "
                "Reformulez avec un sigle (IGGLIA, TEE, TEH, GCA…) ou un métier visé."
            ),
            "information_non_disponible",
            sources=sources,
            sorties=recherche,
        )


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
