# Phase 1 - Analyse exploratoire et contrôle qualité du dataset ML

## Mise à jour V2

La V1 a confirmé une fuite de cible : `parcours_cible` était le `argmax` déterministe d'une fonction utilisant `moyenne_scolaire` et `competences`. La V2 est générée dans `data/full_sample_2000_v2` ; la V1 est conservée dans `data/full_sample_2000` et n'est pas écrasée.

La cible V2 reste une **variable synthétique de simulation**. Elle est tirée après génération du profil selon une utilité latente par parcours : prior synthétique stable, affinités synthétiques avec les intérêts/projets/préférences/environnement, effet faible de la moyenne, contribution limitée des compétences documentées lorsqu'elles existent, puis bruit de type Gumbel non exposé. Les paramètres pour les parcours sans compétence documentée sont des paramètres de simulation, jamais de nouvelles compétences officielles.

Cette méthode est reproductible par seed, mais la cible n'est pas une vérité sur les étudiants réels. Elle ne force pas une distribution uniforme. Avec 2 000 lignes et seed 42, les 16 parcours sont observés ; la classe la plus fréquente compte 185 exemples et la moins fréquente 87. Le ratio est donc d'environ 2,13, moins déséquilibré que la V1.

La règle déterministe V1 ne reconstruit que 143 des 2 000 cibles V2, soit 7,15 %. La fuite exacte de V1 n'est donc plus présente. Cette mesure ne prouve pas l'absence de toute corrélation : elle vérifie seulement que la nouvelle cible n'est pas une copie exacte de l'ancienne règle.

### Limites méthodologiques V2

- Les quatre compétences du corpus sont réutilisées uniquement sous leurs identifiants réels ; elles ne suffisent pas à documenter les 16 parcours.
- Les affinités des parcours sans compétence reliée sont artificielles et doivent être interprétées comme un mécanisme expérimental, pas comme une connaissance institutionnelle.
- Le bruit caché empêche la reconstruction déterministe, mais ne rend pas les profils représentatifs d'étudiants malgaches.
- `projets` reste vide dans 1 218 lignes sur 2 000 (60,90 %), selon la génération synthétique actuelle.
- La couverture des 16 classes est utile pour tester un pipeline, mais ne justifie pas une conclusion sur les formations réelles.

### Résultats V2

| Contrôle | Résultat |
|---|---:|
| Lignes totales | 2 000 |
| Train / validation / test | 1 400 / 300 / 300 |
| Parcours observés | 16 / 16 |
| Doublons | 0 |
| Erreurs de validation | 0 |
| Projets vides | 1 218 (60,90 %) |
| Reconstruction exacte par la règle V1 | 7,15 % |

La décision de générer les 16 parcours est raisonnable uniquement comme simulation technique : les identifiants sont vérifiés par le corpus, mais les parcours insuffisamment documentés ne disposent pas d'une compatibilité institutionnelle défendable. Un entraînement futur devra conserver cette limitation et éviter toute interprétation métier.

## Archive V1 - Analyse initiale

Les sections suivantes conservent les résultats de l'audit V1 afin de garder la trace du problème méthodologique identifié. Elles ne décrivent pas les résultats de la V2.

### Périmètre et reproductibilité V1

L'audit porte sur `data/full_sample_2000` : 2 000 profils synthétiques, seed 42, répartis en 1 400 lignes d'entraînement, 300 de validation et 300 de test. Le notebook reproductible est [notebooks/analyse_exploratoire_dataset.py.ipynb](../notebooks/analyse_exploratoire_dataset.py.ipynb).

Aucun modèle ML n'est entraîné. Le corpus pédagogique officiel n'est pas modifié. Les CSV et le corpus sont lus en lecture seule.

### Qualité générale V1

- 2 000 identifiants candidats, sans doublon.
- Moyennes dans l'intervalle observé 5,03-20,00 ; moyenne 12,898, médiane 12,87, écart-type 3,257.
- Les quatre fichiers de données passent la validation existante (`validation_errors: []`).
- Le split est sans chevauchement d'identifiants et les distributions par split restent observables, mais elles ne sont pas stratifiées explicitement.
- Le schéma contient 16 parcours possibles dans les métadonnées, mais seulement 4 sont effectivement générés.

### Cible `parcours_cible` V1

| Parcours | Nombre | Pourcentage |
|---|---:|---:|
| `parcours-igglia` | 761 | 38,05 % |
| `parcours-esiia` | 557 | 27,85 % |
| `parcours-imticia` | 399 | 19,95 % |
| `parcours-isaia` | 283 | 14,15 % |

La classe majoritaire est `parcours-igglia` ; la classe minoritaire est `parcours-isaia`. Le ratio majoritaire/minoritaire est de 2,69. Le déséquilibre est réel, même s'il n'est pas extrême. Le problème plus important est la couverture : 12 parcours du corpus ne disposent d'aucun exemple.

### Variables V1

### `moyenne_scolaire`

Variable numérique bornée sur 20, simulée par une loi uniforme perturbée par un bruit gaussien puis tronquée. Moyenne 12,898, médiane 12,87, écart-type 3,257, minimum 5,03, maximum 20,00. Par cible, les moyennes sont proches : 12,587 pour IGGLIA, 13,162 pour ESIIA, 13,043 pour IMTICIA et 13,012 pour ISAIA. Elle n'est donc pas le facteur qui explique principalement les classes observées, mais elle participe directement au score de génération.

### `competences`

Les quatre compétences institutionnelles disponibles sont toujours présentes sous forme de scores entiers 0-5. Les moyennes globales sont proches de 3,1 : techniques-informatiques-gestion 3,114 ; électronique-systèmes 3,142 ; multimédia-télécommunications 3,143 ; statistiques-informatique 3,122.

La relation avec la cible est artificiellement forte. Les compétences correspondant au parcours ont une moyenne d'environ 4,62 à 4,71 dans leur classe : IGGLIA 4,623, ESIIA 4,661, IMTICIA 4,694, ISAIA 4,710. Les autres compétences restent autour de 2,2-2,9. Cette séparation est une conséquence directe de la règle de génération.

### `matieres_preferees`

Listes de 1 à 3 items parmi 20 libellés `matiere_synthetique_*`. Il y a 987 chaînes distinctes, avec des répétitions de combinaisons et aucune matière institutionnelle issue du corpus. La variable est donc catégorielle/textuelle mais artificielle et à forte cardinalité.

### `centres_interet`

Listes de 1 à 3 items parmi 9 intérêts synthétiques. Les modalités simples les plus fréquentes sont `management` (87), `sante` (84), `finance` (82), `ecologie` (82) et `robotique` (75). Aucune relation causale avec la cible n'est codée dans le générateur.

### `projets`

Trois états observés : vide (1 181), `projet_personnel_tech` (494), `projet_associatif` (230), ou les deux (95). La variable est vide dans 59,05 % des lignes, car le générateur ne crée un projet technique qu'avec une probabilité de 30 % et un projet associatif avec une probabilité de 15 %. Elle ne participe pas à la cible et ne constitue pas une observation réaliste de projets étudiants.

### `preferences_professionnelles`

Cinq modalités générées par tirage uniforme approximatif : enseignement 417, salariat 415, fonction publique 411, entrepreneuriat 388, recherche 369. La variable n'est pas reliée à la cible par le générateur.

### `environnement_travail`

Quatre modalités générées par tirage uniforme approximatif : hybride 521, bureau 521, télétravail 485, terrain 473. La variable n'est pas reliée à la cible par le générateur.

### Données manquantes V1

| Variable | Valeurs manquantes | Pourcentage |
|---|---:|---:|
| `projets` | 1 181 | 59,05 % |
| Toutes les autres variables | 0 | 0 % |

La valeur vide de `projets` est structurelle et synthétique. Il ne faut pas l'imputer comme si elle représentait une absence réelle déclarée par un étudiant. Avant entraînement, il faut décider si cette variable est retirée, si une modalité explicite `aucun_projet_declare` est conservée, ou si le mécanisme de génération est refondu.

### Risque de fuite de cible V1

Le risque est **bloquant et avéré**. Dans le générateur, chaque cible est choisie par le maximum du score suivant :

- base `moyenne_scolaire / 20` pour tous les parcours ;
- ajout de `0,5 * moyenne(competences du parcours) / 5` lorsque le parcours possède des compétences reliées.

La cible est donc calculée à partir de `moyenne_scolaire` et de `competences`, qui sont ensuite conservées comme variables d'entrée. La reconstruction exacte de la règle sur les 2 000 lignes donne 100 % de concordance. Il s'agit d'une dépendance circulaire potentielle pour un futur modèle : le modèle apprend principalement la règle du générateur, pas une préférence d'orientation observée.

Les 12 parcours sans compétence reliée ne gagnent jamais contre les quatre parcours dotés d'une compétence, puisque la moyenne est commune et que l'ajout de compétence est positif. Cela explique simultanément la couverture incomplète et la concentration sur quatre classes.

### Chaîne de génération V1

```text
profil synthétique
  -> moyenne_scolaire, competences, matières, intérêts, projets,
     préférences professionnelles, environnement de travail
  -> score de compatibilité par parcours
     (moyenne + compétences reliées)
  -> argmax des scores
  -> parcours_cible
```

Variables participant directement à la cible : `moyenne_scolaire` et `competences`. Variables générées mais non utilisées dans la cible : `matieres_preferees`, `centres_interet`, `projets`, `preferences_professionnelles`, `environnement_travail`.

### Biais et limites V1

- **Corpus incomplet :** seules quatre compétences institutionnelles sont disponibles ; elles sont appliquées comme si elles permettaient de distinguer les parcours, alors que le corpus ne fournit pas un référentiel exhaustif parcours par parcours.
- **Biais des compétences disponibles :** les quatre parcours possédant une compétence dominent mécaniquement ; les 12 autres sont impossibles comme cibles.
- **Biais de génération synthétique :** les variables sont tirées de règles simples, indépendantes entre elles, sans validation auprès de vrais étudiants.
- **Déséquilibre :** la distribution 38,05 % / 14,15 % favorise la classe IGGLIA et réduit la couverture de la classe ISAIA.
- **Variables artificielles :** matières et intérêts ne sont pas des observations institutionnelles ou étudiantes ; les préférences professionnelles et l'environnement sont tirés sans relation documentée au parcours.
- **Corrélations artificielles :** la corrélation compétence-cible est créée par construction, et non mesurée sur un résultat réel.
- **Représentativité :** le dataset ne permet pas de prétendre représenter réellement les étudiants malgaches, leurs trajectoires, leurs niveaux, leurs contraintes ou leurs choix.

### Recommandations avant entraînement V1

1. Ne pas entraîner de modèle sur ce dataset dans son état actuel.
2. Définir une cible indépendante des features, issue d'observations ou de scénarios validés, ou retirer les variables utilisées pour fabriquer la cible et reconnaître la limite de cette approche.
3. Compléter et vérifier le référentiel des parcours et compétences avant de générer des classes pour les 16 parcours.
4. Repenser la génération des profils et des variables catégorielles avec des distributions justifiées, puis documenter les hypothèses.
5. Traiter explicitement `projets` et mesurer l'impact de son absence ; ne pas faire d'imputation silencieuse.
6. Régler la couverture et le déséquilibre des classes avant un éventuel entraînement, sans augmenter artificiellement le dataset par simple duplication.
7. Ajouter des contrôles de fuite et de reconstruction de cible dans la validation future.

### Verdict V1

# DATASET À CORRIGER AVANT LE ML

Des corrections importantes sont nécessaires. Elles ne sont pas appliquées automatiquement dans cette phase.
