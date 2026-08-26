"""Audite la V2 avant toute préparation de features ou entraînement ML."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DATASET = RACINE / "data" / "full_sample_2000_v2"
CHEMIN_CORPUS = RACINE / "donnees" / "corpus_pedagogique.json"
FICHIERS_SPLIT = ("train", "val", "test")
COLONNE_CIBLE = "parcours_cible"
COLONNES_PERSONNELLES = {
    "nom",
    "prenom",
    "email",
    "telephone",
    "adresse",
    "date_naissance",
}
MOTIFS_PERSONNELS = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
    re.compile(r"(?<!\d)(?:\+?\d[ .-]?){8,}(?!\d)"),
)


def lire_lignes() -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    """Lit les trois splits sans modifier le dataset."""
    lignes_par_split: dict[str, list[dict[str, str]]] = {}
    toutes_les_lignes: list[dict[str, str]] = []
    for nom_split in FICHIERS_SPLIT:
        chemin = DOSSIER_DATASET / f"{nom_split}.csv"
        with chemin.open(encoding="utf-8", newline="") as fichier:
            lignes = list(csv.DictReader(fichier))
        lignes_par_split[nom_split] = lignes
        toutes_les_lignes.extend(lignes)
    return toutes_les_lignes, lignes_par_split


def moyenne(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def ecart_type(values: list[float]) -> float:
    return round(statistics.pstdev(values), 4) if values else 0.0


def eta_squared(groupes: dict[str, list[float]]) -> float:
    valeurs = [value for groupe in groupes.values() for value in groupe]
    if not valeurs:
        return 0.0
    moyenne_globale = statistics.mean(valeurs)
    variance_totale = sum((value - moyenne_globale) ** 2 for value in valeurs)
    variance_inter = sum(
        len(groupe) * (statistics.mean(groupe) - moyenne_globale) ** 2
        for groupe in groupes.values()
        if groupe
    )
    return round(variance_inter / variance_totale, 4) if variance_totale else 0.0


def information_mutuelle_normalisee(tableau: dict[tuple[str, str], int], total: int) -> float:
    lignes = Counter()
    colonnes = Counter()
    for (ligne, colonne), compte in tableau.items():
        lignes[ligne] += compte
        colonnes[colonne] += compte
    information = 0.0
    for (ligne, colonne), compte in tableau.items():
        if not compte:
            continue
        probabilite = compte / total
        information += probabilite * math.log(
            compte * total / (lignes[ligne] * colonnes[colonne])
        )
    entropie_lignes = -sum(
        (compte / total) * math.log(compte / total) for compte in lignes.values()
    )
    entropie_colonnes = -sum(
        (compte / total) * math.log(compte / total) for compte in colonnes.values()
    )
    denominateur = math.sqrt(entropie_lignes * entropie_colonnes)
    return round(information / denominateur, 4) if denominateur else 0.0


def reconstruire_regle_v1(ligne: dict[str, str], parcours: dict[str, dict]) -> str:
    competences = json.loads(ligne["competences"])
    scores = {}
    for identifiant, definition in parcours.items():
        score = float(ligne["moyenne_scolaire"]) / 20.0
        competences_liees = definition.get("competences", [])
        if competences_liees:
            score += (
                sum(competences.get(cle, 0) for cle in competences_liees)
                / (len(competences_liees) * 5)
                * 0.5
            )
        scores[identifiant] = score
    return max(scores, key=scores.get)


def auditer() -> dict:
    lignes, lignes_par_split = lire_lignes()
    corpus = json.loads(CHEMIN_CORPUS.read_text(encoding="utf-8"))
    identifiants_parcours = {p["identifiant"] for p in corpus["parcours"]}
    parcours = {p["identifiant"]: p for p in corpus["parcours"]}
    colonnes = list(lignes[0])
    entrees = [ligne[colonne] for ligne in lignes for colonne in colonnes]
    valeurs_cibles = set(identifiants_parcours)
    cellules_cible = sum(valeur in valeurs_cibles for valeur in entrees)
    cellules_personnelles = sum(
        motif.search(valeur) is not None for valeur in entrees for motif in MOTIFS_PERSONNELS
    )
    cibles = [ligne[COLONNE_CIBLE] for ligne in lignes]
    groupes_moyenne: dict[str, list[float]] = defaultdict(list)
    groupes_competence: dict[str, list[int]] = defaultdict(list)
    for ligne in lignes:
        groupes_moyenne[ligne[COLONNE_CIBLE]].append(float(ligne["moyenne_scolaire"]))
        competences = json.loads(ligne["competences"])
        for identifiant, valeur in competences.items():
            groupes_competence[f"{ligne[COLONNE_CIBLE]}::{identifiant}"].append(valeur)

    distributions_moyenne = {
        cible: {
            "moyenne": moyenne(valeurs),
            "ecart_type": ecart_type(valeurs),
            "minimum": min(valeurs),
            "maximum": max(valeurs),
        }
        for cible, valeurs in sorted(groupes_moyenne.items())
    }
    distributions_competences = {
        cle: moyenne(valeurs) for cle, valeurs in sorted(groupes_competence.items())
    }
    associations_categorique = {}
    for colonne in (
        "matieres_preferees",
        "centres_interet",
        "projets",
        "preferences_professionnelles",
        "environnement_travail",
    ):
        tableau = Counter((ligne[colonne] or "<vide>", ligne[COLONNE_CIBLE]) for ligne in lignes)
        associations_categorique[colonne] = information_mutuelle_normalisee(tableau, len(lignes))

    reconstructions_v1 = sum(
        reconstruire_regle_v1(ligne, parcours) == ligne[COLONNE_CIBLE] for ligne in lignes
    )
    return {
        "dataset": "data/full_sample_2000_v2",
        "nombre_lignes": len(lignes),
        "splits": {nom: len(valeurs) for nom, valeurs in lignes_par_split.items()},
        "colonnes": colonnes,
        "colonnes_personnelles_detectees": sorted(set(colonnes) & COLONNES_PERSONNELLES),
        "cellules_identifiants_cibles_hors_colonne_cible": cellules_cible - len(cibles),
        "motifs_personnels_detectes": cellules_personnelles,
        "doublons_id_candidat": len(lignes) - len({ligne["id_candidat"] for ligne in lignes}),
        "cibles_inconnues": sorted(set(cibles) - identifiants_parcours),
        "classes": dict(sorted(Counter(cibles).items())),
        "valeurs_vides": {
            colonne: sum(not ligne[colonne] for ligne in lignes) for colonne in colonnes
        },
        "moyenne_par_classe": distributions_moyenne,
        "dependance_moyenne_eta_squared": eta_squared(groupes_moyenne),
        "moyenne_competence_par_classe": distributions_competences,
        "dependance_categorielle_nmi": associations_categorique,
        "reconstruction_exacte_regle_v1": reconstructions_v1,
        "taux_reconstruction_regle_v1": round(reconstructions_v1 / len(lignes), 4),
    }


if __name__ == "__main__":
    print(json.dumps(auditer(), ensure_ascii=False, indent=2))
