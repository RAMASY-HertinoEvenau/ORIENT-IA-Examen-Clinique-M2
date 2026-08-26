"""Générateur de jeux de données synthétiques pour ORIENT'IA (phase 1).

Le module lit le corpus pédagogique vérifié (donnees/corpus_pedagogique.json)
et génère des profils candidats synthétiques, reproductibles et documentés.

Principes:
- Ne modifie pas le corpus.
- Utilise uniquement les identifiants de parcours/compétences réels issus du corpus.
- Toutes les autres variables synthétiques sont clairement marquées comme telles
  dans la documentation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SchemaVariable:
    nom: str
    type: str
    signification: str
    valeurs_possibles: list[str] | None = None
    unite: str | None = None
    utilisation_ml: str | None = None
    provenance: str | None = None


class GenerateurDataset:
    """Générateur déterministe de profils candidats synthétiques.

    Usage simple:

        g = GenerateurDataset(corpus_path)
        g.generer(n=1000, seed=42, out_dir=Path("data/sample"))

    Le générateur produit trois fichiers CSV: train.csv, val.csv, test.csv
    et un fichier metadata.json décrivant le schéma et les seeds utilisées.
    """

    def __init__(self, corpus_path: Path):
        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus introuvable: {self.corpus_path}")

        with self.corpus_path.open(encoding="utf-8") as fh:
            self.corpus = json.load(fh)

        # Extraire les parcours réels du corpus
        self.parcours = [p["identifiant"] for p in self.corpus.get("parcours", [])]
        if not self.parcours:
            raise ValueError("Aucun parcours trouvé dans le corpus; impossible de définir la cible.")

        # Compétences institutionnelles réelles
        self.competences_reelles = [c["identifiant"] for c in self.corpus.get("competences", [])]

        # Variables synthétiques (détail dans la doc)
        self.schema = self._definir_schema()

        # Liste d'intérêts synthétiques (non institutionnels)
        self._interets_simples = [
            "informatique", "management", "ecologie", "sante", "tourisme", "agriculture",
            "design", "finance", "robotique",
        ]

    def _definir_schema(self) -> list[SchemaVariable]:
        # Définition du schéma du profil candidat (noms en français)
        s: list[SchemaVariable] = []
        s.append(SchemaVariable(
            nom="id_candidat",
            type="str",
            signification="Identifiant interne synthétique du candidat",
            utilisation_ml="Index; non utilisé comme feature",
            provenance="synthetique"
        ))

        s.append(SchemaVariable(
            nom="matieres_preferees",
            type="List[str]",
            signification="Liste de matières préférées du candidat (synthétique)",
            valeurs_possibles=None,
            utilisation_ml="Feature textuelle/catégorielle (embeddings ou one-hot)",
            provenance="synthetique - pas de matières institutionnelles dans le corpus"
        ))

        s.append(SchemaVariable(
            nom="moyenne_scolaire",
            type="float",
            signification="Moyenne générale sur 20 lors du dernier diplôme obtenu",
            unite="/20",
            utilisation_ml="Feature numérique (normalisation)",
            provenance="synthetique - simulée"
        ))

        s.append(SchemaVariable(
            nom="competences",
            type="Dict[str,int]",
            signification="Score synthétique (0-5) pour chaque compétence; clés = identifiants de compétences institutionnelles quand disponibles",
            utilisation_ml="Features numériques pour chaque compétence institutionnelle",
            provenance="variable synthétique de simulation; clés limitées aux compétences du corpus"
        ))

        s.append(SchemaVariable(
            nom="centres_interet",
            type="List[str]",
            signification="Centres d'intérêt personnels (liste synthétique)",
            utilisation_ml="Feature textuelle/catégorielle",
            provenance="synthetique"
        ))

        s.append(SchemaVariable(
            nom="projets",
            type="List[str]",
            signification="Projets personnels ou scolaires (mots-clés synthétiques)",
            utilisation_ml="Feature textuelle",
            provenance="synthetique"
        ))

        s.append(SchemaVariable(
            nom="preferences_professionnelles",
            type="str",
            signification="Secteur ou rôle professionnel privilégié",
            valeurs_possibles=["entrepreneuriat", "salariat", "recherche", "enseignement", "fonction_publique"],
            utilisation_ml="Feature catégorielle",
            provenance="synthetique"
        ))

        s.append(SchemaVariable(
            nom="environnement_travail",
            type="str",
            signification="Préférence pour le mode de travail",
            valeurs_possibles=["bureau", "terrain", "teletravail", "hybride"],
            utilisation_ml="Feature catégorielle",
            provenance="synthetique"
        ))

        s.append(SchemaVariable(
            nom="parcours_cible",
            type="str",
            signification="Identifiant du parcours choisi (cible ML)",
            valeurs_possibles=self.parcours,
            utilisation_ml="Cible (label) - une seule valeur parmi les parcours du corpus",
            provenance="variable synthétique de simulation; identifiant issu du corpus"
        ))

        return s

    def _sample_competences(self, rng: random.Random) -> dict[str, int]:
        # Les scores sont simulés; aucun référentiel de compétence n'est enrichi.
        d: dict[str, int] = {}
        for comp in self.competences_reelles:
            # Distribution simple: scores entiers de 0 à 5.
            p = rng.random()
            if p < 0.05:
                score = 0
            elif p < 0.7:
                score = rng.randint(1, 4)
            else:
                score = 5
            d[comp] = score
        return d

    @staticmethod
    def _stable_unit(*parts: str) -> float:
        payload = "|".join(parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def _score_simulation(
        self,
        parcours: dict,
        moyenne: float,
        competences: dict[str, int],
        centres: list[str],
        projets: list[str],
        preference: str,
        environnement: str,
    ) -> float:
        """Calcule une utilité latente, non officielle, pour le choix simulé."""
        pid = parcours["identifiant"]
        score = -0.25 + 0.5 * self._stable_unit("prior", pid)
        score += (moyenne - 12.5) / 20.0 * (self._stable_unit("grade", pid) - 0.5)
        for interest in centres:
            score += (self._stable_unit("interest", pid, interest) - 0.5) * 0.35
        for project in projets:
            score += (self._stable_unit("project", pid, project) - 0.5) * 0.15
        score += (self._stable_unit("preference", pid, preference) - 0.5) * 0.3
        score += (self._stable_unit("environment", pid, environnement) - 0.5) * 0.2
        for competence in parcours.get("competences", []):
            score += competences.get(competence, 0) / 5.0 * 0.12
        return score

    def _tirer_parcours_cible(
        self,
        rng: random.Random,
        moyenne: float,
        competences: dict[str, int],
        centres: list[str],
        projets: list[str],
        preference: str,
        environnement: str,
    ) -> str:
        """Tire une cible selon une utilité softmax perturbée par un bruit caché."""
        utilities = [
            self._score_simulation(
                parcours,
                moyenne,
                competences,
                centres,
                projets,
                preference,
                environnement,
            )
            for parcours in self.corpus.get("parcours", [])
        ]
        temperature = 0.9
        noisy_utilities = []
        for utility in utilities:
            uniform = max(rng.random(), 1e-12)
            gumbel_noise = -math.log(-math.log(uniform)) * temperature
            noisy_utilities.append(utility + gumbel_noise)
        index = max(range(len(noisy_utilities)), key=noisy_utilities.__getitem__)
        return self.parcours[index]

    def generer(self, n: int, seed: int, repartition: tuple[float, float, float] = (0.7, 0.15, 0.15), out_dir: Path | None = None) -> dict[str, Path]:
        """Génère `n` profils et écrit train/val/test dans `out_dir`.

        - Reproductible via `seed`.
        - `repartition` doit être tuple (train, val, test) somme==1.
        """
        if sum(repartition) <= 0:
            raise ValueError("Répartition invalide")

        rng = random.Random(seed)

        rows = []
        for i in range(n):
            cid = f"cand-{i:06d}"
            # matières préférées: synthétique, 1-3 items
            nb_mat = rng.randint(1, 3)
            mat_pref = [f"matiere_synthetique_{rng.randint(1,20)}" for _ in range(nb_mat)]

            moyenne = round(rng.uniform(8.0, 18.0) + rng.gauss(0, 1.5), 2)
            if moyenne < 0:
                moyenne = 0.0
            if moyenne > 20:
                moyenne = 20.0

            competences = self._sample_competences(rng)

            centres = rng.sample(self._interets_simples, rng.randint(1, 3))

            projets = []
            if rng.random() < 0.3:
                projets.append("projet_personnel_tech")
            if rng.random() < 0.15:
                projets.append("projet_associatif")

            prefs = rng.choice(["entrepreneuriat", "salariat", "recherche", "enseignement", "fonction_publique"])
            env = rng.choice(["bureau", "terrain", "teletravail", "hybride"])

            cible = self._tirer_parcours_cible(
                rng,
                moyenne,
                competences,
                centres,
                projets,
                prefs,
                env,
            )

            row = {
                "id_candidat": cid,
                "matieres_preferees": ";".join(mat_pref),
                "moyenne_scolaire": moyenne,
                "competences": json.dumps(competences, ensure_ascii=False),
                "centres_interet": ";".join(centres),
                "projets": ";".join(projets),
                "preferences_professionnelles": prefs,
                "environnement_travail": env,
                "parcours_cible": cible,
            }
            rows.append(row)

        # split
        n_total = len(rows)
        idxs = list(range(n_total))
        rng.shuffle(idxs)
        n_train = int(repartition[0] * n_total)
        n_val = int(repartition[1] * n_total)
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val:]

        if out_dir is None:
            out_dir = Path("data/generated")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def write_csv(path: Path, indices: list[int]):
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for i in indices:
                    writer.writerow(rows[i])

        train_p = out_dir / "train.csv"
        val_p = out_dir / "val.csv"
        test_p = out_dir / "test.csv"

        write_csv(train_p, train_idx)
        write_csv(val_p, val_idx)
        write_csv(test_p, test_idx)

        # metadata
        meta = {
            "dataset_version": "v2",
            "generation_method": "utilite_synthetique_softmax_bruit_cache",
            "simulation_warning": "La cible est synthetique et ne represente pas une verite sur des etudiants reels.",
            "n_total": n_total,
            "repartition": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "seed": seed,
            "schema": [asdict(x) for x in self.schema],
        }
        meta_p = out_dir / "metadata.json"
        with meta_p.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        return {"train": train_p, "val": val_p, "test": test_p, "meta": meta_p}

    def valider_dataset(self, paths: dict[str, Path]) -> list[str]:
        """Effectue des contrôles qualité simples et retourne la liste des erreurs trouvées.

        Vérifications:
        - identifiants de parcours existants
        - valeurs hors plage pour `moyenne_scolaire`
        - doublons d'`id_candidat`
        - format JSON pour la colonne `competences`
        """
        errors: list[str] = []
        seen_ids = set()

        def check_file(p: Path):
            with p.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            for r in rows:
                cid = r.get("id_candidat")
                if cid in seen_ids:
                    errors.append(f"Doublon id_candidat: {cid} dans {p}")
                seen_ids.add(cid)

                try:
                    moy = float(r.get("moyenne_scolaire", ""))
                except (ValueError, TypeError):
                    errors.append(f"Moyenne invalide pour {cid} dans {p}")
                    continue
                if not (0.0 <= moy <= 20.0):
                    errors.append(f"Moyenne hors plage pour {cid}: {moy}")

                comp_raw = r.get("competences", "")
                try:
                    _ = json.loads(comp_raw) if comp_raw else {}
                except json.JSONDecodeError:
                    errors.append(f"Competences JSON invalide pour {cid} dans {p}")

                try:
                    competences = json.loads(comp_raw) if comp_raw else {}
                except json.JSONDecodeError:
                    competences = {}
                if set(competences) != set(self.competences_reelles):
                    errors.append(f"Ensemble de competences invalide pour {cid} dans {p}")
                if any(not isinstance(value, int) or not 0 <= value <= 5 for value in competences.values()):
                    errors.append(f"Score de competence hors plage pour {cid} dans {p}")

                # target verification
                cible = r.get("parcours_cible")
                if cible not in self.parcours:
                    errors.append(f"Parcours cible inconnu pour {cid}: {cible}")

        for key in ("train", "val", "test"):
            check_file(paths[key])

        return errors
