"""Génère un jeu de données synthétique et produit des statistiques d'audit.

Usage:
    python scripts/generate_and_audit.py --n 2000 --seed 42 --out data/full_sample_2000
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from orient_ia.generateur_dataset import GenerateurDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/full_sample_2000")
    args = p.parse_args()

    corpus = Path("donnees/corpus_pedagogique.json")
    g = GenerateurDataset(corpus)
    paths = g.generer(args.n, seed=args.seed, out_dir=Path(args.out))

    rows = []
    for split in ("train", "val", "test"):
        pth = paths[split]
        with pth.open(encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))
        rows.extend([(split, r) for r in reader])

    n_total = len(rows)
    counts = {"train": sum(1 for s, _ in rows if s == "train"),
              "val": sum(1 for s, _ in rows if s == "val"),
              "test": sum(1 for s, _ in rows if s == "test")}

    labels = [r["parcours_cible"] for _, r in rows]
    class_dist = Counter(labels)

    fields = list(rows[0][1].keys()) if rows else []
    missing = {}
    for f in fields:
        missing[f] = sum(1 for _, r in rows if (r.get(f, "") in (None, "", "[]", "{}")))

    ids = [r["id_candidat"] for _, r in rows]
    duplicates = len(ids) - len(set(ids))

    errors = g.valider_dataset(paths)

    stats = {
        "n_total": n_total,
        "counts": counts,
        "class_distribution": dict(class_dist),
        "missing_counts": missing,
        "duplicates": duplicates,
        "validation_errors": errors,
        "paths": {k: str(v) for k, v in paths.items()},
        "seed": args.seed,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(json.dumps({"status": "ok", "n_total": n_total, "duplicates": duplicates, "errors": len(errors)}))


if __name__ == "__main__":
    main()
