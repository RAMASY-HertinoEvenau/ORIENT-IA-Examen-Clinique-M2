"""Vérifications de sécurité et de robustesse pour l'agent ORIENT'IA."""
from __future__ import annotations

from collections.abc import Iterable

_MOTS_INJECTION = {
    "ignore previous instructions",
    "ignore all rules",
    "do not follow",
    "bypass",
    "override system",
    "pretend to be",
    "secret prompt",
    "developer mode",
    "jailbreak",
}

_MOTS_SENSIBLES = {
    "score",
    "admission",
    "note",
    "réussite",
    "décision finale",
    "validation",
    "valider",
    "recrutement",
    "recruter",
    "sexe",
    "âge",
    "date de naissance",
    "nationalité",
}


def valider_message(message: str) -> tuple[bool, str | None]:
    """Retourne un message d'erreur s'il y a une demande hors périmètre."""
    texte = (message or "").lower().strip()
    if not texte:
        return False, "Aucun message n'a été fourni."

    if any(mot in texte for mot in _MOTS_INJECTION):
        return False, "Je ne peux pas exécuter des demandes qui visent à contourner les règles de sécurité ou les consignes du système."

    if any(mot in texte for mot in _MOTS_SENSIBLES):
        return False, "Je peux aider à orienter la personne, mais je ne peux pas prendre de décision administrative ou juger des informations sensibles comme l'âge ou d'autres données personnelles."

    return True, None


def extraire_termes_liste(valeur: object) -> list[str]:
    if valeur is None:
        return []
    if isinstance(valeur, str):
        return [valeur.strip()] if valeur.strip() else []
    if isinstance(valeur, Iterable) and not isinstance(valeur, (str, bytes, dict)):
        return [str(item).strip() for item in valeur if str(item).strip()]
    return [str(valeur).strip()] if str(valeur).strip() else []
