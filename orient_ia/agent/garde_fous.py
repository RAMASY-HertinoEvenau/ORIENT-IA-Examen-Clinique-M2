"""Module de sécurité et de filtrage des garde-fous éthiques et techniques."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte).encode("ASCII", "ignore").decode("utf-8")
    return texte.lower()


@dataclass
class ReponseSecurite:
    bloque: bool
    motif: Optional[str] = None
    reponse_alternative: Optional[str] = None


class AnalyseurSecurite:
    """Analyseur pour la prévention des injections de prompt, des biais et des profilages interdits."""

    MOTS_INJECTION = [
        r"ignore.*(consigne|document|source|instruction|officiel)",
        r"oublie.*(regle|instruction|consigne)",
        r"fais comme si",
        r"jailbreak",
        r"mode bypass",
        r"system prompt",
        r"affirme qu.*nouvelle filiere",
    ]

    MOTS_PROFILAGE_PSYCHO = [
        r"analyse.*(personnalite|caractere|profil psychologique|traits de personnalite)",
        r"d.apres mes messages.*personnalite",
        r"psycho",
        r"mbti",
        r"temperament",
    ]

    MOTS_DISCRIMINATION = [
        r"recommande.*(uniquement|selon|d.apres).*(sexe|genre|femme|homme|age|ethnie|race|origine)",
        r"(parcours|filiere).*(reservee?|destinee?|specifique|pour).*(fille|garcon|homme|femme)",
        r"(filiere|parcours).*(femme|homme|fille|garcon)",
        r"selon (mon|le) sexe",
    ]

    MOTS_ADMINISTRATIF = [
        r"(je suis|suis-je) (admis|accepte|selectionne|recu)",
        r"(valide|confirme) mon admission",
        r"decision officielle",
    ]

    def analyser_message(self, message: str) -> ReponseSecurite:
        msg_norm = _normaliser(message)

        # 1. Détection de Prompt Injection
        for motif in self.MOTS_INJECTION:
            if re.search(motif, msg_norm):
                return ReponseSecurite(
                    bloque=True,
                    motif="prompt_injection",
                    reponse_alternative=(
                        "ORIENT'IA applique des consignes strictes de traçabilité. "
                        "Je ne peux ignorer les sources officielles ni affirmer l'existence de formations non publiées par l'ISPM."
                    ),
                )

        # 2. Refus de profilage psychologique (Section 16 du sujet)
        for motif in self.MOTS_PROFILAGE_PSYCHO:
            if re.search(motif, msg_norm):
                return ReponseSecurite(
                    bloque=True,
                    motif="refus_profilage_psychologique",
                    reponse_alternative=(
                        "ORIENT'IA ne procède à aucun profilage psychologique ni inférence de traits de personnalité à partir de vos messages. "
                        "Les recommandations pédagogiques reposent exclusivement sur les critères scolaires, compétences et centres d'intérêt que vous déclarez explicitement."
                    ),
                )

        # 3. Refus de critères discriminatoires (Section 16 du sujet)
        for motif in self.MOTS_DISCRIMINATION:
            if re.search(motif, msg_norm):
                return ReponseSecurite(
                    bloque=True,
                    motif="refus_discrimination",
                    reponse_alternative=(
                        "Les recommandations d'orientation de l'ISPM sont strictement fondées sur les prérequis académiques, "
                        "les aptitudes et les projets professionnels. Le sexe, l'âge ou toute autre caractéristique sensible ne constituent en aucun cas des critères de recommandation valides."
                    ),
                )

        # 4. Clarification administrative
        for motif in self.MOTS_ADMINISTRATIF:
            if re.search(motif, msg_norm):
                return ReponseSecurite(
                    bloque=True,
                    motif="clarification_administrative",
                    reponse_alternative=(
                        "ORIENT'IA constitue un outil indicatif d'aide à l'orientation. "
                        "Ses réponses ne remplacent en aucun cas une décision officielle d'admission, qui relève exclusivement de la commission de sélection de l'ISPM après dépôt de dossier."
                    ),
                )

        return ReponseSecurite(bloque=False)
