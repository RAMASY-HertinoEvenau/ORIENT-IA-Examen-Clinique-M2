"""Compare les baselines et modèles simples sur la validation V2 uniquement."""
from __future__ import annotations

import json

from orient_ia.ml.experimentations import executer_experiences


def main() -> None:
    resultats = executer_experiences()
    resume = [
        {
            "experience": resultat["experience"],
            "type": resultat["type"],
            "f1_macro": resultat["f1_macro"],
            "balanced_accuracy": resultat["balanced_accuracy"],
            "accuracy": resultat["accuracy"],
            "log_loss": resultat.get("log_loss"),
        }
        for resultat in resultats
    ]
    print(json.dumps(resume, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
