# ORIENT'IA — Assistant Intelligent d'Aide à l'Orientation Pédagogique

Prototype d'assistant d'orientation développé pour l'Institut Supérieur Polytechnique de Madagascar (ISPM) dans le cadre de l'examen de fin d'études de Master 2.

---

## 1. Principe & Architecture

ORIENT'IA combine l'apprentissage statistique supervisé (Machine Learning), la recherche documentaire augmentée par la génération (RAG) et une couche d'orchestration conversationnelle sous contraintes de sécurité et de traçabilité.

```text
donnees/       Corpus vérifié ISPM, registre des sources et données d'enquête réelle
data/          Jeux de données d'entraînement synthétiques contrôlés (anti-fuite)
ml/            Artefacts sérialisés du modèle ExtraTrees et métadonnées
orient_ia/
  ├── rag/     Moteur de recherche hybride et traçabilité des passages
  ├── agent/   Orchestrateur, outils fonctionnels et garde-fous de sécurité
  ├── ml/      Featurisation, validation croisée et modèles
  └── api/     Serveur FastAPI / Endpoints REST
interface/     Interface Web utilisateur avec suivi de traçabilité et chat
evaluation/    Jeu de 32 cas de test officiels (9 catégories réglementaires)
scripts/       Scripts reproductibles d'entraînement et d'évaluation globale
documentation/ Rapports d'évaluation, architecture, registre et analyse des risques
```

---

## 2. Installation & Démarrage rapide

### Prérequis
- Python 3.11+
- PowerShell ou terminal Bash

### Installation de l'environnement virtuel

```powershell
# 1. Création et activation de l'environnement
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Installation des dépendances
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 3. Exécution de l'Application

### Étape 1 : Démarrer le Serveur Backend API
```powershell
uvicorn orient_ia.api.serveur:app --reload --port 8000
```
*L'API est alors disponible sur `http://localhost:8000` (documentation Swagger interactive sur `http://localhost:8000/docs`).*

### Étape 2 : Lancer l'Interface Web Utilisateur
Ouvrez simplement le fichier `interface/index.html` dans votre navigateur Web moderne (Chrome, Firefox, Edge).
L'interface se connecte automatiquement au backend `http://localhost:8000`.

---

## 4. Tests et Évaluation Expérimentale

### Exécuter la suite complète de tests unitaires et d'intégration
```powershell
pytest
```

### Lancer le Benchmark Officiel des 32 cas de test
```powershell
python scripts/evaluer_systeme_complet.py
```
*Génère le rapport chiffré dans `documentation/rapport_evaluation_globale.md`.*

---

## 5. Mention obligatoire
> **À garder en tête :** ORIENT’IA constitue un outil d’aide à l’orientation. Ses recommandations ne remplacent ni l’avis d’un conseiller pédagogique ni une décision officielle d’admission.
