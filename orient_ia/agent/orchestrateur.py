"""Orchestration de l'agent conversationnel ORIENT'IA."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orient_ia.agent.modeles import TraceConversation
from orient_ia.agent.outils import analyser_profil, comparer_parcours, rechercher_formations
from orient_ia.agent.securite import valider_message

_CHAMPS_OBLIGATOIRES = ("matieres_preferees", "moyenne_scolaire", "competences")


def _normaliser_profil(profil: Mapping[str, Any] | None) -> dict[str, Any]:
    if profil is None:
        return {}
    return dict(profil)


def _champs_manquants(profil: Mapping[str, Any]) -> list[str]:
    manquants: list[str] = []
    for champ in _CHAMPS_OBLIGATOIRES:
        if champ not in profil or profil.get(champ) in (None, [], {}, ""):
            manquants.append(champ)
    return manquants


def _message_autour_recommandation(profil: Mapping[str, Any], conseil: dict[str, Any]) -> str:
    recommandations = conseil.get("recommandations", [])
    if not recommandations:
        return "Je n’ai pas assez de signal pour proposer une recommandation fiable. Merci de compléter le profil."

    items = []
    for parcours in recommandations[:3]:
        nom = parcours.get("nom") or parcours.get("formation") or "Parcours"
        score = parcours.get("score")
        items.append(f"- {nom}{f' ({score})' if score is not None else ''}")

    return (
        "Voici les pistes les plus cohérentes avec le profil fourni :\n"
        + "\n".join(items)
        + "\n\nJe reste prudent : la recommandation aide l’orientation, elle ne remplace pas une décision humaine."
    )


def _inspire_question_suivante(profil: Mapping[str, Any]) -> str:
    manquants = _champs_manquants(profil)
    if not manquants:
        return "Je peux maintenant calculer une recommandation structurée à partir de votre profil."

    labels = {
        "matieres_preferees": "quelles sont vos matières préférées ?",
        "moyenne_scolaire": "quelle est votre moyenne scolaire ?",
        "competences": "quelles compétences ou points forts identifiez-vous ?",
    }
    return "Avant de recommander un parcours, merci de me donner " + "; ".join(labels.get(champ, champ) for champ in manquants) + "."


def orchestrer_conversation(message: str, profil: Mapping[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
    """Point d'entrée principal : traite la demande, sécurise l'entrée et orchestre les outils."""
    ok, erreur = valider_message(message)
    if not ok:
        return {
            "reponse": erreur,
            "etat": "refuse",
            "outils_appeles": [],
            "sources": [],
            "trace": TraceConversation(session_id=session_id, message=message, etat="refuse", reponse=erreur).to_dict(),
        }

    profil_normalise = _normaliser_profil(profil)
    texte = (message or "").strip()
    trace = TraceConversation(session_id=session_id, message=texte, etat="analyse", donnees_profil=dict(profil_normalise))

    if "comparer" in texte.lower() or "comparaison" in texte.lower():
        trace.outils_appeles.append("comparer_parcours")
        mots_nettoyes = (
            texte.lower()
            .replace("comparer", "")
            .replace("comparaison", "")
            .replace("les parcours", "")
            .replace("parcours", "")
            .replace("du", "")
            .replace("de", "")
            .replace("des", "")
            .replace("et", ",")
            .replace("vs", ",")
        )
        candidats = [p.strip() for p in mots_nettoyes.split(",") if p.strip() and p.strip() not in ("le", "la", "les", "?")]
        if not candidats and "informatique" in texte.lower():
            candidats = ["informatique", "marketing"]

        if candidats:
            comparaison = comparer_parcours(candidats)
        else:
            comparaison = {"statut": "incomplet", "message": "Je peux comparer plusieurs parcours si vous me donnez leurs noms ou identifiants."}

        trace.etat = "comparaison"
        trace.reponse = comparaison.get("message", "Comparaison indisponible.")
        trace.sources = comparaison.get("resultat", [])
        return {"reponse": trace.reponse, "etat": trace.etat, "outils_appeles": trace.outils_appeles, "sources": trace.sources, "trace": trace.to_dict()}

    est_demande_recommandation = any(k in texte.lower() for k in ("recommand", "conseil", "conseille", "propose-moi", "proposer", "option est la meilleure", "que me conseillez"))
    mots_recherche_stricte = ("formations", "accès", "condition", "compétence", "diplôme", "niveaux")

    if est_demande_recommandation or (profil_normalise and not any(k in texte.lower() for k in mots_recherche_stricte)):
        champs = _champs_manquants(profil_normalise)
        if champs or not profil_normalise:
            trace.etat = "besoin_informations"
            trace.reponse = _inspire_question_suivante(profil_normalise)
            return {"reponse": trace.reponse, "etat": trace.etat, "outils_appeles": trace.outils_appeles, "sources": [], "trace": trace.to_dict()}

        trace.outils_appeles.append("analyser_profil")
        extrait = analyser_profil(profil_normalise)
        conseil = extrait.get("resultat", {})
        trace.etat = "recommandation"
        trace.reponse = _message_autour_recommandation(profil_normalise, conseil)
        trace.sources = conseil.get("sources", []) if isinstance(conseil, dict) else []
        return {"reponse": trace.reponse, "etat": trace.etat, "outils_appeles": trace.outils_appeles, "sources": trace.sources, "trace": trace.to_dict()}

    trace.outils_appeles.append("rechercher_formations")
    resultats = rechercher_formations(texte, nombre_resultats=3, seuil=0.05)
    trace.etat = "recherche"
    trace.reponse = resultats.get("message", "Aucune information documentaire pertinente n'a été trouvée.")
    trace.sources = resultats.get("resultats", [])
    return {"reponse": trace.reponse, "etat": trace.etat, "outils_appeles": trace.outils_appeles, "sources": trace.sources, "trace": trace.to_dict()}


# Alias fonctionnel attendu par le code API.
def traiter_message(message: str, profil: Mapping[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
    return orchestrer_conversation(message=message, profil=profil, session_id=session_id)
