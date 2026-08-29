# Architecture du Système ORIENT'IA

## 1. Vue d'ensemble du Système

ORIENT'IA est un système hybride combinant l'apprentissage statistique (Machine Learning), la recherche d'information augmentée par la génération (RAG) et une couche d'orchestration conversationnelle sous contraintes éthiques et réglementaires.

```mermaid
flowchart TD
    subgraph "1. Sources & Données"
        A1[Corpus Institutionnel ISPM<br/>Site web & Fiches d'accès] --> B1[Registre des Sources<br/>& Découpage en Passages]
        A2[Dataset Synthétique Contrôlé<br/>2000 profils / Anti-fuite] --> B2[Pipeline Featurisation ML]
        A3[Enquête Personnes Réelles<br/>Étudiants & Professionnels] --> B3[Registre de Collecte<br/>& Réponses Anonymisées]
    end

    subgraph "2. Couche Intelligence (ML & RAG)"
        B1 --> C1[Moteur RAG Hybride<br/>TF-IDF + BM25 + Citations]
        B2 --> C2[Modèle ExtraTrees<br/>100 arbres / Classifieur]
    end

    subgraph "3. Agent Orchestrateur & Sécurité"
        D1[Garde-fous de Sécurité<br/>Anti-injection / Anti-profilage psycho] --> E1[Orchestrateur Agent ORIENT'IA]
        C1 --> E1
        C2 --> E1
        E2[Outils Dédiés<br/>Recherche / Vérif Prérequis / Comparaison / Scoring ML] <--> E1
        E1 --> E3[Moteur de Traçabilité & Observabilité<br/>Latence, Outils, E/S ML, Sources]
    end

    subgraph "4. Interface & Déploiement"
        E1 --> F1[Serveur API FastAPI<br/>Endpoints /profil/analyser & /agent/message]
        F1 --> G1[Interface Utilisateur Web<br/>Formulaire profil, Chat, Volet de traçabilité]
        E1 --> G2[Benchmark d'Évaluation<br/>32 cas de test / Rapport chiffré]
    end
```

---

## 2. Description des Composants

### 2.1 Moteur RAG et Traçabilité (`orient_ia/rag/`)
- Ingestion du corpus vérifié `donnees/corpus_pedagogique.json`.
- Indexation vectorielle hybride (n-grammes TF-IDF et filtrage sémantique).
- Extraction systématique de la source de provenance (URL, date de consultation, statut institutionnel).
- Gestion stricte de l'absence d'information : toute mention ou matière non officielle est signalée comme absente.

### 2.2 Composant Machine Learning (`orient_ia/ml/`)
- Modèle de classification supervisé `ExtraTreesClassifier` entraîné avec protocole strict anti-fuite.
- Transformation reproductible des variables textuelles et catégorielles.
- Sortie probabiliste calibrée et calcul de score d'adéquation indicatif accompagné de son niveau d'incertitude.

### 2.3 Agent Conversationnel et Garde-fous (`orient_ia/agent/`)
- **Outils identifiables :** `rechercher_formation`, `analyser_profil_ml`, `verifier_prerequis`, `comparer_parcours`.
- **Garde-fous éthiques :**
  - Blocage des tentatives de détournement de consignes (prompt injections).
  - Refus formel du profilage psychologique et de l'inférence de traits de personnalité.
  - Refus strict de tout critère discriminatoire (sexe, âge).
  - Clarification administrative : rappel que seul le conseil de sélection de l'ISPM valide les admissions.

### 2.4 API Backend & Interface Utilisateur (`orient_ia/api/` & `interface/`)
- API RESTful FastAPI fournissant les services en temps réel.
- Interface web ergonomique avec mention légale obligatoire, formulaire de profilage, chat interactif et volet dépliable d'observabilité.
