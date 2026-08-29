# Note sur les Limites, les Biais et les Risques — ORIENT'IA

Ce document formalise l'analyse critique des limites méthodologiques, des biais de données et des risques techniques et éthiques associés au prototype ORIENT'IA, conformément aux exigences du sujet d'examen de Master 2.

---

## 1. Limites des Données et du Corpus

1. **Périmètre documentaire partiel :**
   - Le corpus pédagogique officiel extrait du site de l'ISPM le 26 août 2026 ne comporte pas la maquette détaillée des cours semestre par semestre ni la liste exhaustive des passerelles inter-filières.
   - Ces manques sont conservés tels quels et explicités à l'utilisateur : aucune extrapolation non vérifiée n'est faite.
2. **Données d'entraînement synthétiques :**
   - Le jeu de données principal (2000 profils) est généré selon des règles probabilistes contrôlées. Bien qu'efficace pour apprendre la structure relationnelle entre intérêts et parcours, il ne saurait remplacer un historique académique pluriannuel complet.
3. **Échantillon de l'enquête réelle :**
   - L'enquête de validation (122 répondants) présente des intervalles de confiance larges en raison de la taille d'échantillon par parcours.

---

## 2. Biais Identifiés et Mesures de Mitigation

| Biais identifié | Description du risque | Mesure de mitigation appliquée |
| :--- | :--- | :--- |
| **Biais d'auto-sélection** | Surreprésentation des répondants issus des filières informatiques dans l'enquête. | Pondération équilibrée dans les règles sémantiques et déclaration de l'incertitude. |
| **Biais de reconstruction** | Chez les professionnels, souvenir rétrospectif idéalisé des motivations initiales. | Séparation de la validation entre cohorte étudiante (choix présent) et cohorte professionnelle (adéquation métier). |
| **Biais de genre / stéréotypes** | Risque d'orienter arbitrairement selon le sexe ou l'âge. | **Suppression totale** des variables sensibles dans les features ML et blocage de sécurité automatique des requêtes discriminatoires. |
| **Biais d'autorité / Hallucination** | Risque que le LLM affirme une fausse règle d'admission. | Ancrage RAG strict sur les sources officielles et refus d'inventer des formations. |

---

## 3. Risques Techniques, Éthiques et Garde-fous

1. **Injections de prompt et manipulation :**
   - L'analyseur de sécurité intercepte les tentatives de détournement de consignes (ex: *« ignore les règles officielles »*) avant toute exécution d'outil.
2. **Refus du profilage psychologique (Règle d'or Section 16) :**
   - Le système s'interdit d'inférer des traits de personnalité (ex: MBTI, tempérament) à partir du style rédactionnel. Seuls les faits déclarés explicitement par l'utilisateur sont pris en compte.
3. **Confusion entre conseil et décision administrative :**
   - L'interface et les réponses de l'agent rappellent systématiquement la mention légale : **ORIENT'IA est un outil indicatif d'aide à l'orientation et ne valide pas d'admission officielle.**

---

## 4. Observabilité et Traçabilité

Pour chaque recommandation, l'intégralité du chemin décisionnel est journalisée :
- Question utilisateur brute et profil déclaré.
- Passages documentaires extraits et scores de similarité.
- Outils appelés avec leurs paramètres.
- Entrées et sorties du modèle de Machine Learning.
- Temps d'exécution (latence moyenne < 15 ms).
- Motif de refus ou d'alerte en cas de requête non conforme.
