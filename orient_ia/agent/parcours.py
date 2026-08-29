"""Identification des parcours ISPM et formulation des réponses conversationnelles."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple


def normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte).encode("ASCII", "ignore").decode("utf-8")
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", " ", texte)
    return " ".join(texte.split())


# Alias du plus spécifique au plus générique. Les termes trop vagues (ex. "informatique")
# ne sont pas mappés à un unique parcours.
ALIAS_PARCOURS: Dict[str, Tuple[str, ...]] = {
    "parcours-igglia": (
        "igglia",
        "informatique de gestion",
        "genie logiciel",
        "developpement logiciel",
    ),
    "parcours-isaia": (
        "isaia",
        "informatique statistique",
        "statistique appliquee",
        "data science",
    ),
    "parcours-esiia": (
        "esiia",
        "systeme informatique et intelligence",
        "electronique systeme",
        "informatique embarquee",
        "systemes embarques",
    ),
    "parcours-imticia": (
        "imticia",
        "multimedia",
        "telecommunications",
        "telecom",
    ),
    "parcours-emii": (
        "emii",
        "electro mecanique",
        "electromecanique",
        "informatique industrielle",
    ),
    "parcours-icmp": (
        "icmp",
        "industries chimiques",
        "minieres",
        "petrolieres",
        "petrole",
    ),
    "parcours-gca": (
        "gca",
        "genie civil",
        "architecture",
        "btp",
    ),
    "parcours-caa": (
        "caa",
        "commerce et administration",
        "administration des affaires",
    ),
    "parcours-emp": (
        "emp",
        "economie et management",
        "management de projet",
    ),
    "parcours-fic": (
        "fic",
        "finances et comptabilites",
        "comptabilite",
        "finance",
    ),
    "parcours-dtja": (
        "dtja",
        "droit et techniques",
        "juridiques des affaires",
        "droit des affaires",
    ),
    "parcours-iaa": (
        "iaa",
        "agroalimentaire",
        "industrie agroalimentaire",
    ),
    "parcours-aee": (
        "aee",
        "agriculture et elevage",
        "agronomie",
        "elevage",
    ),
    "parcours-pip": (
        "pip",
        "pharmacologie",
        "pharmaceutique",
        "pharmacie",
    ),
    "parcours-tee": (
        "tee",
        "tourisme et environnement",
        "ecotourisme",
        "tourisme environnement",
    ),
    "parcours-teh": (
        "teh",
        "tourisme et hotellerie",
        "hotellerie",
        "restauration",
        "hebergement",
    ),
}

NOMS_COURTS = {
    "parcours-igglia": "IGGLIA",
    "parcours-isaia": "ISAIA",
    "parcours-esiia": "ESIIA",
    "parcours-imticia": "IMTICIA",
    "parcours-emii": "EMII",
    "parcours-icmp": "ICMP",
    "parcours-gca": "GCA",
    "parcours-caa": "CAA",
    "parcours-emp": "EMP",
    "parcours-fic": "FIC",
    "parcours-dtja": "DTJA",
    "parcours-iaa": "IAA",
    "parcours-aee": "AEE",
    "parcours-pip": "PIP",
    "parcours-tee": "TEE",
    "parcours-teh": "TEH",
}

PAIRES_PROCHES = {
    "parcours-tee": "parcours-teh",
    "parcours-teh": "parcours-tee",
    "parcours-igglia": "parcours-isaia",
    "parcours-isaia": "parcours-igglia",
    "parcours-esiia": "parcours-emii",
    "parcours-emii": "parcours-esiia",
    "parcours-caa": "parcours-emp",
    "parcours-emp": "parcours-caa",
    "parcours-iaa": "parcours-aee",
    "parcours-aee": "parcours-iaa",
}


def extraire_codes_parcours(texte: str) -> List[str]:
    """Retourne les identifiants de parcours cités, dans l'ordre d'apparition."""
    norm = normaliser(texte)
    trouves: List[Tuple[int, int, str]] = []
    deja = set()

    for code, aliases in ALIAS_PARCOURS.items():
        meilleur: Optional[Tuple[int, int]] = None
        for alias in aliases:
            motif = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            match = re.search(motif, norm)
            if not match:
                continue
            candidat = (match.start(), len(alias))
            if meilleur is None or candidat[1] > meilleur[1] or (
                candidat[1] == meilleur[1] and candidat[0] < meilleur[0]
            ):
                meilleur = candidat
        if meilleur is not None:
            trouves.append((meilleur[0], -meilleur[1], code))
            deja.add(code)

    trouves.sort()
    return [code for _, __, code in trouves]


def extraire_serie_bacc(texte: str) -> Optional[str]:
    norm = normaliser(texte)
    if re.search(r"\b(serie|bacc?|baccalaureat)?\s*a2\b", norm) or re.search(r"\bbacc? a\b", norm):
        return "A2"
    if re.search(r"\b(serie|bacc?|baccalaureat)?\s*c\b", norm) or "bacc c" in norm:
        return "C"
    if re.search(r"\b(serie|bacc?|baccalaureat)?\s*d\b", norm) or "bacc d" in norm:
        return "D"
    if re.search(r"\b(serie|bacc?|baccalaureat)?\s*s\b", norm) or "bacc s" in norm:
        return "S"
    if "technique industrielle" in norm or "tech ind" in norm:
        return "TECHNIQUE INDUSTRIELLE"
    if "technique agricole" in norm:
        return "TECHNIQUE AGRICOLE"
    return None


def detecter_intention(texte: str) -> str:
    norm = normaliser(texte)
    codes = extraire_codes_parcours(texte)
    comparaison = any(
        m in norm
        for m in ("difference", "comparer", "comparaison", "versus", "ecart entre", "ou choisir")
    ) or " vs " in f" {norm} "

    if comparaison or (len(codes) >= 2 and re.search(r"\bou\b", norm)):
        return "comparaison"
    if len(codes) >= 2:
        return "comparaison"
    if any(m in norm for m in ("metier", "debouche", "emploi", "carriere", "travail apres")):
        return "metiers"
    if any(m in norm for m in ("matiere", "programme", "enseignement", "cours")):
        return "matieres"
    if any(m in norm for m in ("competence", "apprend", "savoir faire")):
        return "competences"
    if any(m in norm for m in ("bacc", "serie", "admissib", "prerequis", "condition d acces", "inscription", "admission")):
        return "prerequis"
    if any(m in norm for m in ("meilleure filiere", "meilleur parcours", "meilleure formation", "classement", "laquelle est la meilleure")):
        return "ambiguite"
    if any(m in norm for m in ("recommand", "orienter", "quel parcours", "que faire", "conseil", "propose")):
        return "recommandation"
    if len(codes) == 1:
        return "fiche"
    return "recherche"


def sigle(code: str, nom: str = "") -> str:
    if code in NOMS_COURTS:
        return NOMS_COURTS[code]
    match = re.search(r"\(([A-Z]{2,8})\)", nom)
    return match.group(1) if match else code.replace("parcours-", "").upper()


def fiche_depuis_passage(passage: Any) -> Dict[str, Any]:
    meta = getattr(passage, "metadata", {}) or {}
    code = meta.get("id") or getattr(passage, "identifiant", "")
    nom = meta.get("nom") or getattr(passage, "titre", code)
    return {
        "code": code,
        "sigle": sigle(code, nom),
        "titre": getattr(passage, "titre", nom),
        "nom": nom,
        "contenu": getattr(passage, "contenu", ""),
        "competences": list(meta.get("competences") or []),
        "descriptions_competences": list(meta.get("descriptions_competences") or []),
        "matieres": list(meta.get("matieres") or []),
        "prerequis": list(meta.get("prerequis") or []),
        "metiers": list(meta.get("metiers") or []),
        "categorie": getattr(passage, "categorie", meta.get("type", "")),
    }


def _puces(items: Sequence[str], repli: str) -> str:
    propres = [str(x).strip() for x in items if str(x).strip()]
    if not propres:
        return f"- {repli}"
    return "\n".join(f"- {item}" for item in propres)


def formater_fiche(fiche: Dict[str, Any], intention: str = "fiche") -> str:
    nom = fiche.get("nom") or fiche.get("titre") or "ce parcours"
    sig = fiche.get("sigle") or ""
    titre = f"{nom}" if sig and sig in nom else f"{nom} ({sig})" if sig else nom

    competences = fiche.get("competences") or []
    descriptions = fiche.get("descriptions_competences") or []
    orientation = descriptions[0] if descriptions else (competences[0] if competences else "Formation professionnelle de l'ISPM.")
    metiers = fiche.get("metiers") or []
    matieres = fiche.get("matieres") or []
    prerequis = fiche.get("prerequis") or []

    if intention == "metiers":
        return (
            f"Après **{titre}**, les débouchés recensés dans les présentations officielles ISPM sont :\n\n"
            f"{_puces(metiers, 'Débouchés du secteur, non détaillés nommément dans le corpus.')}\n\n"
            f"Ces métiers restent indicatifs : l'insertion dépend du parcours, des stages et du marché."
        )
    if intention == "competences":
        return (
            f"**{titre}** vise principalement :\n\n"
            f"{_puces(competences, 'Compétences professionnelles du département.')}\n\n"
            f"{orientation}"
        )
    if intention == "matieres":
        if matieres:
            corps = _puces(matieres, "")
        else:
            corps = (
                "- Le détail des matières par semestre n'est pas publié dans les sources ISPM indexées.\n"
                "- La formation combine des enseignements fondamentaux et de spécialité du département."
            )
        return f"Pour **{titre}** :\n\n{corps}"
    if intention == "prerequis":
        return (
            f"Conditions d'accès publiées pour **{titre}** :\n\n"
            f"{_puces(prerequis, 'Être titulaire du baccalauréat et passer la sélection sur dossier.')}\n\n"
            f"L'admission définitive reste une décision de l'administration de l'ISPM."
        )

    blocs_comp = _puces(competences, "Compétences professionnelles du département.")
    blocs_metiers = _puces(metiers, "Métiers du secteur d'activité.")
    blocs_acces = _puces(prerequis, "Baccalauréat et sélection sur dossier.")
    return (
        f"**{titre}**\n\n"
        f"**Orientation :** {orientation}\n\n"
        f"**Compétences visées :**\n{blocs_comp}\n\n"
        f"**Débouchés :**\n{blocs_metiers}\n\n"
        f"**Accès :**\n{blocs_acces}"
    )


def formater_comparaison(fiche_a: Dict[str, Any], fiche_b: Dict[str, Any]) -> str:
    sa, sb = fiche_a.get("sigle") or "Parcours A", fiche_b.get("sigle") or "Parcours B"
    na, nb = fiche_a.get("nom") or sa, fiche_b.get("nom") or sb

    def axe(fiche: Dict[str, Any]) -> str:
        desc = (fiche.get("descriptions_competences") or [""])[0]
        comps = fiche.get("competences") or []
        return desc or (comps[0] if comps else "non précisé dans le corpus")

    def metiers_courts(fiche: Dict[str, Any]) -> str:
        mets = fiche.get("metiers") or []
        if not mets:
            return "débouchés du secteur, non listés nommément"
        if len(mets) == 1:
            return mets[0]
        return f"{mets[0]} ; {mets[1]}" + ("…" if len(mets) > 2 else "")

    return (
        f"### {sa} et {sb} : ce n'est pas la même orientation\n\n"
        f"**{na}**\n"
        f"- Logique de formation : {axe(fiche_a)}\n"
        f"- Exemples de métiers : {metiers_courts(fiche_a)}\n\n"
        f"**{nb}**\n"
        f"- Logique de formation : {axe(fiche_b)}\n"
        f"- Exemples de métiers : {metiers_courts(fiche_b)}\n\n"
        f"**En clair :** {sa} prépare plutôt à « {axe(fiche_a)} », "
        f"tandis que {sb} prépare plutôt à « {axe(fiche_b)} ». "
        f"Le choix dépend de ce que vous voulez exercer au quotidien, pas d'un classement des filières."
    )


def formater_series(serie: str, detail_parcours: Optional[str] = None) -> str:
    serie = serie.upper()
    lignes = [
        f"Avec un **Bac {serie}**, l'ISPM publie les règles suivantes (sélection sur dossier, pas d'admission automatique) :",
        "",
        "- **Informatique, Télécommunication et Génie Industriel** : séries C, D, S et techniques industrielles.",
        "- **Biotechnologie et Agronomie** : C, D, S, techniques agricoles, et A2 si la note de maths est au moins 12.",
        "- **Techniques des Affaires et Tourisme** (dont TEE et TEH) : toutes séries.",
        "- **Autres mentions** : accès en 1re année pour tout titulaire du baccalauréat, sous réserve de la sélection du dossier.",
    ]
    if detail_parcours:
        lignes.extend(["", detail_parcours])
    else:
        lignes.extend(["", "Indiquez une filière (ex. IGGLIA, GCA, TEE) pour une réponse ciblée."])
    return "\n".join(lignes)
