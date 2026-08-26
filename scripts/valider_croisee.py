"""Exécute la validation croisée Phase 2.3 sur train.csv uniquement."""

import json

from orient_ia.ml.validation_croisee import executer_validation_croisee

if __name__ == "__main__":
    resultat = executer_validation_croisee()
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
