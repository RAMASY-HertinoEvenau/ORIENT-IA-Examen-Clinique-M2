"""Évalue les trois candidats sur val.csv uniquement."""
import json

from orient_ia.ml.validation_independante import executer_validation_independante

if __name__ == "__main__":
    resultat = executer_validation_independante()
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
