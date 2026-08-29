# Phase 2.3 - Validation indépendante sur val.csv

## Protocole

Cette étape a utilisé uniquement `data/full_sample_2000_v2/train.csv` et `val.csv`.

- Seed fixe : 42.
- Featuriseur ajusté sur train uniquement.
- Modèle ajusté sur train uniquement.
- Validation indépendante réalisée une seule fois sur val.
- Hyperparamètres inchangés depuis l'expérimentation précédente.
- Aucun accès à `test.csv`.
- Corpus, générateur et CSV V1/V2 inchangés.

Les trois candidats sont exactement ceux conservés après la validation croisée : Extra Trees sans compétences, SVM linéaire sans compétences et Extra Trees sans moyenne.

## Résultats sur val.csv

| Candidat | F1 macro | Balanced accuracy | Accuracy | Précision macro | Rappel macro | Classes à rappel nul | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Extra Trees sans compétences** | **0,094799** | **0,097321** | **0,110000** | **0,097171** | **0,097321** | **3** | 4,310875 |
| Extra Trees sans moyenne | 0,084643 | 0,086612 | 0,103333 | 0,085481 | 0,086612 | 5 | 3,499741 |
| SVM linéaire sans compétences | 0,071737 | 0,074727 | 0,083333 | 0,079864 | 0,074727 | 4 | n/a |

Les 16 classes sont présentes dans la matrice de confusion et les métriques par classe. La matrice complète et les rappels détaillés sont produits directement par `executer_validation_independante` et ne sont pas stockés dans les données sources.

### Classes à rappel nul

- Extra Trees sans compétences : `parcours-aee`, `parcours-caa`, `parcours-emii`.
- Extra Trees sans moyenne : `parcours-emii`, `parcours-iaa`, `parcours-igglia`, `parcours-imticia`, `parcours-teh`.
- SVM linéaire sans compétences : `parcours-dtja`, `parcours-iaa`, `parcours-icmp`, `parcours-igglia`.

Le rappel nul reste un signal d'alerte. Il ne permet pas à lui seul de déclarer un modèle invalide puisque le problème concerne les trois candidats, mais il interdit de présenter le système comme fiable pour toutes les classes.

## Comparaison selon le protocole

Le critère principal est le F1 macro. Les écarts avec Extra Trees sans compétences sont :

- contre Extra Trees sans moyenne : `+0,010156` en F1 macro et `+0,010709` en balanced accuracy ;
- contre SVM linéaire sans compétences : `+0,023062` en F1 macro et `+0,022594` en balanced accuracy.

La différence avec Extra Trees sans moyenne est inférieure au seuil méthodologique de `0,02` sur les deux métriques principales. Elle ne suffit donc pas à départager ces deux candidats à elle seule.

La différence avec le SVM dépasse `0,02` sur val, mais la validation croisée précédente montrait des moyennes très proches :

- Extra Trees sans compétences : F1 macro `0,068166`, balanced accuracy `0,068995` ;
- SVM linéaire sans compétences : F1 macro `0,068008`, balanced accuracy `0,074564` ;
- Extra Trees sans moyenne : F1 macro `0,067484`, balanced accuracy `0,069519`.

La différence val contre le SVM doit donc être interprétée avec prudence. Elle ne suffit pas à effacer la variabilité observée entre folds et seeds, surtout avec 16 classes et seulement 300 observations de validation.

## Décision méthodologique

Extra Trees sans compétences est le meilleur candidat sur cette validation indépendante, mais il n'est pas suffisamment supérieur à Extra Trees sans moyenne pour être figé de manière robuste selon la règle des `0,02`. Il présente aussi trois classes à rappel nul.

Le SVM linéaire est moins bon sur val, mais sa balanced accuracy moyenne en validation croisée était la meilleure des candidats. Aucun modèle ne domine donc simultanément la validation croisée et la validation indépendante avec une marge robuste sur tous les critères.

**Aucun candidat ne satisfait suffisamment les critères de sélection.** Extra Trees sans compétences est le meilleur résultat ponctuel sur `val.csv`, mais son avance sur Extra Trees sans moyenne est insuffisante : `+0,010156` en F1 macro et `+0,010709` en balanced accuracy, soit moins que le seuil de `0,02`. Les résultats de validation croisée ne permettent pas non plus de départager robustement les candidats : les moyennes sont très proches et le SVM obtenait la meilleure balanced accuracy moyenne.

Les rappels nuls persistants constituent une limite importante : aucun candidat ne garantit une reconnaissance correcte des 16 classes. Les performances observées sont faibles, instables et mesurées exclusivement sur des données synthétiques. Elles ne doivent donner lieu à aucune prétention de performance, de fiabilité métier ou d'efficacité pour l'orientation d'étudiants réels.

Les candidats restent conservés pour la suite méthodologique, mais aucun accès à `test.csv` n'est autorisé sur la base de cette comparaison seule.

## Test final

`test.csv` n'a pas été chargé et doit rester totalement indépendant. Avant toute utilisation du test, il faudrait :

1. décider explicitement si la priorité absolue est le F1 macro ou la balanced accuracy ;
2. documenter le maintien ou l'abandon des candidats proches ;
3. figer un seul modèle et ses hyperparamètres avant lecture du test ;
4. réentraîner ce modèle selon un protocole annoncé à l'avance ;
5. utiliser `test.csv` une seule fois comme mesure finale ;
6. ne plus modifier le modèle après observation du test.

Cette étape n'effectue aucune de ces opérations finales.

## Limites

Les scores sont faibles et concernent exclusivement un dataset synthétique. Ils ne mesurent pas une capacité d'orientation réelle et ne représentent pas les étudiants malgaches. Les classes à rappel nul, la petite validation et la génération synthétique rendent la sélection fragile.
