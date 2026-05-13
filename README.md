# ORIENT'IA

Prototype d'assistant intelligent d'aide a l'orientation pedagogique, realise dans le cadre de l'examen de Master 2 a l'ISPM.

## Principe

ORIENT'IA combine un profil declare par l'utilisateur, un classement ML, des informations documentaires tracees et des regles pedagogiques. Le systeme devra produire une recommandation argumentee, prudente et accompagnee de son niveau d'incertitude. Il ne prendra jamais une decision officielle d'admission et ne presentera pas une information non verifiee comme officielle.

## Architecture

```text
donnees/       Sources, donnees brutes et donnees traitees
ml/            Preparation, entrainement, evaluation et artefacts ML
rag/           Ingestion, indexation et recherche documentaire
agent/         Outils, regles et orchestration conversationnelle
domaine/       Modeles et services metier
api/           Frontiere HTTP (phase ulterieure)
interface/     Interface de demonstration (phase ulterieure)
tests/         Tests unitaires et d'integration
notebooks/     Analyse exploratoire reproductible
evaluation/    Jeu de cas et rapports d'evaluation
scripts/       Commandes reproductibles
documentation/ Decisions, methodologie et limites
```

## Corpus pédagogique Phase 1

Le corpus vérifié est disponible dans
[donnees/corpus_pedagogique.json](donnees/corpus_pedagogique.json) et son
périmètre est décrit dans
[documentation/phase-1-corpus.md](documentation/phase-1-corpus.md). Il contient
uniquement les mentions, parcours, niveaux, compétences, conditions d'accès et
le débouché ISAIA effectivement publiés par l'ISPM le 26 août 2026.

Les matières principales, les passerelles et les débouchés non publiés restent
absents. Les matières de concours de l'annuaire externe ne sont pas utilisées
comme matières de formation. L'habilitation ministérielle annoncée par l'ISPM
n'a pas été vérifiée par un acte officiel indépendant.

## Installation locale

Python 3.11 ou plus recent est requis.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```
