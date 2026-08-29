# Phase 2.3 - Baselines et expérimentation ML contrôlée

## Périmètre

Cette étape utilise exclusivement `data/full_sample_2000_v2`. Le fichier `test.csv` n'est pas chargé par l'expérimentation : il reste réservé à une évaluation finale après comparaison et choix documenté. Le corpus, le générateur et les CSV V2 n'ont pas été modifiés. Aucun commit n'est créé dans cette étape.

L'implémentation est dans [orient_ia/ml/experimentations.py](../orient_ia/ml/experimentations.py), le lanceur dans [scripts/evaluer_baselines_modeles.py](../scripts/evaluer_baselines_modeles.py), et les tests dans [tests/test_experimentations.py](../tests/test_experimentations.py).

## Protocole

Pour chaque configuration :

1. lecture de `train.csv` et `val.csv` uniquement ;
2. séparation de `parcours_cible` et `id_candidat` ;
3. ajustement d'un nouveau featuriseur sur train ;
4. transformation de train et validation avec ce featuriseur ;
5. apprentissage sur train ;
6. comparaison sur validation.

Les cinq ablations existantes sont utilisées pour les trois modèles : toutes les variables, sans moyenne, sans compétences, sans variables textuelles et variables synthétiques uniquement. Aucune recherche d'hyperparamètres n'est effectuée.

## Baselines

- **Classe majoritaire :** prédit toujours la classe la plus fréquente du train.
- **Aléatoire stratifiée :** tire les classes selon les proportions observées dans train, avec seed 42.
- **Moyenne seule :** classifieur par centroïde de la moyenne standardisée par classe.
- **Compétences seules :** classifieur par centroïde des quatre scores de compétences.
- **Variables synthétiques uniquement :** classifieur par centroïde sur textes et catégories synthétiques, sans moyenne ni compétences.

Les centroïdes sont des baselines simples, sans optimisation ni modèle de production.

## Modèles testés

- régression logistique multinomiale (`max_iter=1000`) ;
- SVM linéaire ;
- Extra Trees (`100` arbres, seed 42, un seul processus).

## Métriques

Pour chaque expérience, le code calcule :

- F1 macro ;
- balanced accuracy ;
- accuracy ;
- précision par classe ;
- rappel par classe ;
- matrice de confusion avec les 16 classes ;
- log loss lorsque des probabilités sont disponibles.

Les résultats sont calculés sur validation uniquement. Les performances ne sont pas des performances sur test et ne représentent pas des étudiants réels.

## Résultats validation

| Expérience | F1 macro | Balanced accuracy | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| Classe majoritaire | 0,013060 | 0,062500 | 0,116667 | 31,838560 |
| Aléatoire stratifiée | 0,088367 | 0,092223 | 0,086667 | 2,756226 |
| Centroïdes, moyenne seule | 0,031322 | 0,068772 | 0,080000 | n/a |
| Centroïdes, compétences seules | 0,036180 | 0,042750 | 0,046667 | n/a |
| Centroïdes, variables synthétiques | 0,084756 | 0,098798 | 0,096667 | n/a |
| Régression logistique, toutes variables | 0,069618 | 0,071510 | 0,076667 | 2,894642 |
| SVM linéaire, toutes variables | 0,065737 | 0,070786 | 0,080000 | n/a |
| Extra Trees, toutes variables | 0,084799 | 0,090319 | 0,100000 | 3,704897 |
| Régression logistique, sans moyenne | 0,068835 | 0,071774 | 0,083333 | 2,881705 |
| SVM linéaire, sans moyenne | 0,055394 | 0,058093 | 0,070000 | n/a |
| Extra Trees, sans moyenne | 0,084643 | 0,086612 | 0,103333 | 3,499741 |
| Régression logistique, sans compétences | 0,063148 | 0,067650 | 0,073333 | 2,870123 |
| SVM linéaire, sans compétences | 0,071737 | 0,074727 | 0,083333 | n/a |
| **Extra Trees, sans compétences** | **0,094799** | **0,097321** | **0,110000** | 4,310875 |
| Régression logistique, sans variables textuelles | 0,058663 | 0,067799 | 0,080000 | 2,794390 |
| SVM linéaire, sans variables textuelles | 0,048043 | 0,066439 | 0,080000 | n/a |
| Extra Trees, sans variables textuelles | 0,051369 | 0,054373 | 0,056667 | 10,654844 |
| Régression logistique, variables synthétiques | 0,063512 | 0,068626 | 0,076667 | 2,857502 |
| SVM linéaire, variables synthétiques | 0,062984 | 0,068576 | 0,080000 | n/a |
| Extra Trees, variables synthétiques | 0,086441 | 0,086751 | 0,100000 | 4,965803 |

Les matrices de confusion et les précisions/rappels par classe sont calculés et retournés par `executer_experiences`. Toutes les expériences utilisent le même ensemble de 16 classes du train.

## Lecture des résultats

Les scores sont faibles, ce qui est cohérent avec une cible V2 probabiliste, synthétique et bruitée. La classe majoritaire n'est pas une référence suffisante : elle atteint 11,67 % d'accuracy mais seulement 1,31 % de F1 macro.

Le meilleur F1 macro observé est obtenu par Extra Trees sans compétences (`0,094799`), devant la baseline aléatoire stratifiée (`0,088367`). Cette différence est faible et n'est pas encore une preuve de supériorité robuste. Elle ne justifie pas une optimisation supplémentaire à ce stade.

Le modèle initialement considéré **provisoirement pour une éventuelle évaluation finale** était Extra Trees sans compétences, uniquement parce qu'il obtenait le meilleur F1 macro et la meilleure accuracy sur la première seed. Ce choix est réévalué dans la section multi-seeds ci-dessous et n'est finalement pas confirmé.

Aucune évaluation finale sur `test.csv` n'a été lancée.

## Analyse comparative multi-seeds

Une comparaison supplémentaire a été réalisée avec les seeds 11, 22, 33, 44 et 55, toujours sur train/validation uniquement. Les transformations ont été réajustées sur train pour chaque expérience ; `test.csv` n'a pas été chargé.

| Expérience | F1 macro moyen ± écart-type | Balanced accuracy moyenne ± écart-type | Accuracy moyenne ± écart-type |
|---|---:|---:|---:|
| Aléatoire stratifiée | 0,060935 ± 0,012508 | 0,062537 ± 0,012329 | 0,067333 ± 0,011431 |
| Extra Trees, toutes variables | 0,076686 ± 0,014940 | 0,080228 ± 0,015640 | 0,092000 ± 0,017333 |
| Extra Trees, sans moyenne | 0,078261 ± 0,012420 | **0,083580 ± 0,012396** | 0,094000 ± 0,011814 |
| Extra Trees, sans compétences | 0,077534 ± 0,013520 | 0,078501 ± 0,014074 | 0,090667 ± 0,016248 |
| Extra Trees, sans variables textuelles | 0,043834 ± 0,006233 | 0,046834 ± 0,006286 | 0,049333 ± 0,007118 |
| Extra Trees, variables synthétiques uniquement | **0,081513 ± 0,009788** | 0,081801 ± 0,009573 | 0,092667 ± 0,010625 |

La baseline par centroïdes avec variables synthétiques uniquement reste à F1 macro 0,084756 et balanced accuracy 0,098798, mais elle n'est pas directement comparable à un modèle Extra Trees et ne constitue pas une preuve de supériorité robuste.

### Classes et matrice de confusion

Pour Extra Trees sans compétences avec seed 42, les classes `parcours-aee`, `parcours-caa` et `parcours-emii` ont un rappel nul sur la validation. Les rappels les plus faibles sont aussi observés pour `parcours-isaia` (0,0476), `parcours-teh` (0,0526), `parcours-fic` (0,0556) et `parcours-dtja` (0,0588). Les meilleurs rappels restent faibles : `parcours-emp` 0,3043, `parcours-tee` 0,2222 et `parcours-esiia` 0,2000.

La matrice de confusion est très diffuse : sur 16 classes, les erreurs sont réparties entre de nombreuses destinations. Les confusions les plus nombreuses observées pour cette seed sont `isaia -> esiia` (8), `gca -> imticia` (5), puis plusieurs confusions à 4 observations, notamment `teh -> igglia`, `teh -> gca`, `pip -> esiia`, `pip -> emp` et `imticia -> tee`. Cela ne montre pas une frontière de classes nette.

La baseline aléatoire obtient un F1 macro moyen de 0,0609. Extra Trees sans compétences lui est supérieur de seulement 0,0166 point en moyenne, avec un écart-type de 0,0135. La différence est donc du même ordre que la variabilité de l'expérience et ne suffit pas à établir une supériorité statistique.

### Influence des groupes de variables

- **Moyenne scolaire :** retirer la moyenne augmente légèrement le F1 moyen Extra Trees (0,0783 contre 0,0767 avec toutes les variables) et le balanced accuracy (0,0836 contre 0,0802). La moyenne n'apporte donc pas un signal stable dans cette V2.
- **Compétences :** retirer les compétences ne donne pas de gain robuste : F1 moyen 0,0775 contre 0,0767 avec toutes les variables, et balanced accuracy 0,0785 contre 0,0802. Leur influence apparente est faible.
- **Variables textuelles :** les retirer dégrade fortement Extra Trees (F1 moyen 0,0438). Cette relation doit être interprétée avec prudence, car les textes sont synthétiques et `matieres_preferees` est à forte cardinalité.
- **Variables synthétiques :** Extra Trees avec variables synthétiques uniquement obtient le meilleur F1 moyen (0,0815), mais ce résultat reflète le mécanisme de génération synthétique et ne prouve aucune pertinence pédagogique.

### Décision comparative

Le choix initial « Extra Trees sans compétences » n'est pas confirmé. Il n'est ni le meilleur en F1 macro moyen, ni le meilleur en balanced accuracy moyen, et ses performances par classe sont insuffisantes et instables.

**Conclusion : les différences sont trop faibles et trop sensibles à la seed pour justifier le choix clair d'un modèle.** Aucun modèle ne doit être retenu pour une évaluation finale sur `test.csv` à ce stade. Une évaluation finale ne devra être autorisée qu'après décision méthodologique explicite sur le critère de sélection, ou après une nouvelle étape de validation autorisée.

## Contrôles anti-fuite

Les tests vérifient :

- `parcours_cible` absent des features finales ;
- séparation de train, validation et test ;
- absence de lecture de `test.csv` pendant l'expérimentation ;
- reproductibilité avec seed fixe ;
- fonctionnement des baselines ;
- présence des 16 classes dans l'évaluation ;
- transformation apprise par le train uniquement via le featuriseur existant.

## Limites

- Le dataset est entièrement synthétique.
- La cible ne représente pas un choix réel d'étudiant.
- Les 16 parcours ne sont pas tous documentés par des compétences institutionnelles détaillées.
- Les textes, projets et catégories sont synthétiques.
- `projets` est vide dans 60,90 % des lignes.
- La faible taille de validation par classe rend les comparaisons instables.
- Une validation supérieure à une baseline ne démontre aucune utilité métier.
- Les probabilités et la calibration restent exploratoires.

## État

La Phase 2.3 est limitée aux baselines et à la comparaison train/validation. La préparation de l'évaluation finale sur test, toute sélection définitive et toute interprétation métier nécessitent une étape d'autorisation distincte.
