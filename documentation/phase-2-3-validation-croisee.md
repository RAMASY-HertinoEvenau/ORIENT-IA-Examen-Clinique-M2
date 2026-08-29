# Phase 2.3 - Validation croisée avant évaluation finale

## Protocole exécuté

La validation croisée a utilisé exclusivement `data/full_sample_2000_v2/train.csv`.

- Validation croisée stratifiée : 5 folds.
- Seeds : 42, 43 et 44.
- Évaluations par expérience : 15 (3 seeds × 5 folds).
- Transformations : nouveau `FeaturiseurML` ajusté dans chaque fold sur la partie apprentissage uniquement.
- `parcours_cible` : séparé avant la featurisation.
- `id_candidat` : exclu avant la featurisation.
- `val.csv` : non utilisé par cette étape.
- `test.csv` : non chargé.
- Corpus, générateur et CSV V1/V2 : inchangés.

Le code est dans [orient_ia/ml/validation_croisee.py](../orient_ia/ml/validation_croisee.py) et le lanceur dans [scripts/valider_croisee.py](../scripts/valider_croisee.py). Les résultats détaillés par fold et seed sont retournés par `executer_validation_croisee` ; aucun fichier de résultats n'est conservé dans le dépôt.

## Expériences

Baselines comparées : classe majoritaire, aléatoire stratifiée, centroïdes sur moyenne seule, centroïdes sur compétences seules et centroïdes sur variables synthétiques.

Candidats conservés :

- Extra Trees sans compétences ;
- Extra Trees sans moyenne ;
- Extra Trees avec variables synthétiques uniquement ;
- SVM linéaire sans compétences.

## Résultats agrégés

Les intervalles sont des intervalles de confiance à 95 % calculés sur les 15 scores fold/seed. Ils décrivent la dispersion expérimentale, sans constituer une preuve de généralisation à des étudiants réels.

| Expérience | F1 macro moyen | IC 95 % | Balanced accuracy moyenne | IC 95 % | Accuracy moyenne | Classes à rappel nul moyen / max |
|---|---:|---:|---:|---:|---:|---:|
| Classe majoritaire | 0,010246 | [0,010246 ; 0,010246] | 0,062500 | [0,062500 ; 0,062500] | 0,089286 | 15,0 / 15 |
| Aléatoire stratifiée | 0,066089 | [0,057989 ; 0,074188] | 0,066732 | [0,058502 ; 0,074962] | 0,068809 | 4,87 / 9 |
| Centroïdes, moyenne | 0,030550 | [0,025743 ; 0,035357] | 0,072501 | [0,067118 ; 0,077884] | 0,075000 | 11,8 / 14 |
| Centroïdes, compétences | 0,046093 | [0,040341 ; 0,051845] | 0,059240 | [0,052116 ; 0,066364] | 0,057857 | 8,07 / 12 |
| Centroïdes, synthétiques | 0,064093 | [0,057757 ; 0,070429] | 0,072671 | [0,066971 ; 0,078371] | 0,071190 | 5,2 / 9 |
| Extra Trees, sans compétences | 0,068166 | [0,060736 ; 0,075596] | 0,068995 | [0,061503 ; 0,076487] | 0,073333 | 5,0 / 9 |
| Extra Trees, sans moyenne | 0,067484 | [0,059542 ; 0,075425] | 0,069519 | [0,062129 ; 0,076909] | 0,075714 | 5,4 / 9 |
| Extra Trees, synthétiques | 0,064527 | [0,057502 ; 0,071553] | 0,065071 | [0,058078 ; 0,072064] | 0,067143 | 4,93 / 7 |
| SVM linéaire, sans compétences | **0,068008** | [0,060308 ; 0,075708] | **0,074564** | [0,066188 ; 0,082940] | **0,084286** | 5,93 / 8 |

## Classement selon le protocole

Le critère principal défini précédemment est le F1 macro moyen. En cas d'écart inférieur à 0,02, la différence est considérée comme faible et la balanced accuracy, les rappels par classe et la stabilité départagent seulement à titre descriptif.

1. Extra Trees sans compétences : F1 macro 0,068166.
2. SVM linéaire sans compétences : F1 macro 0,068008.
3. Extra Trees sans moyenne : F1 macro 0,067484.
4. Extra Trees variables synthétiques uniquement : F1 macro 0,064527.

L'écart entre le premier et le deuxième est de 0,000158 ; l'écart entre le premier et le troisième est de 0,000682. Ces différences sont très inférieures au seuil de 0,02 et les intervalles se recouvrent largement. Le SVM linéaire sans compétences obtient toutefois les meilleures balanced accuracy et accuracy moyennes parmi les candidats.

## Rappels par classe

Aucun candidat ne garantit un rappel positif pour toutes les classes dans chaque fold. Le rappel nul est fréquent : même la meilleure configuration sur ce critère, Extra Trees avec variables synthétiques uniquement, compte en moyenne 4,93 classes à rappel nul et jusqu'à 7 sur un fold/seed.

La classe majoritaire prédit uniquement une classe et laisse 15 classes à rappel nul. La baseline aléatoire présente aussi plusieurs rappels nuls par fold. Ce problème est lié à la petite taille des folds par classe, au bruit de la cible synthétique et à l'absence de signaux pédagogiques suffisamment documentés. Il doit être signalé, mais ne constitue pas un veto isolé puisque toutes les configurations sont concernées.

## Conclusion méthodologique

La validation croisée ne confirme pas le choix initial d'Extra Trees sans compétences comme meilleur modèle robuste. Son F1 moyen est à peine supérieur à celui du SVM et d'Extra Trees sans moyenne, mais son balanced accuracy moyenne est inférieure à celle du SVM.

Les différences sont trop faibles pour justifier une sélection définitive. Plusieurs candidats doivent rester ouverts :

- Extra Trees sans compétences, meilleur F1 moyen de manière marginale ;
- SVM linéaire sans compétences, meilleur balanced accuracy et meilleure accuracy ;
- Extra Trees sans moyenne, très proche des deux précédents.

Aucun modèle ne peut encore être autorisé pour l'évaluation finale sur `val.csv` sans décision supplémentaire. L'étape suivante devra comparer ces candidats sur `val.csv` avec le protocole figé, puis choisir un seul modèle avant tout accès à `test.csv`.

## Contrôles

Les tests associés vérifient :

- 5 folds et 3 seeds ;
- 15 résultats par expérience ;
- présence des 16 classes ;
- moyenne, écart-type, intervalle de confiance et résultats fold/seed ;
- absence de chargement de `test.csv` ;
- reproductibilité ;
- séparation des variables interdites.

## Limites

Cette validation concerne uniquement la robustesse interne d'un mécanisme synthétique. Elle ne mesure ni la pertinence pédagogique, ni la représentativité d'étudiants malgaches, ni une performance en production. Les scores faibles et les rappels nuls doivent être rapportés comme résultats négatifs ou exploratoires, sans les transformer en promesse métier.
