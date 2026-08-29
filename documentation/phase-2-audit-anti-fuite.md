# Phase 2 - Protocole expérimental et audit anti-fuite

## Périmètre

Cet audit constitue uniquement l'étape 1 de la Phase 2. Il utilise exclusivement `data/full_sample_2000_v2`, seed 42. Les fichiers V1, le corpus pédagogique et les CSV V2 n'ont pas été modifiés.

Aucun modèle ML n'a été entraîné. Aucune préparation de features n'a été réalisée. Le contrôle reproductible est implémenté dans [scripts/auditer_phase2_anti_fuite.py](../scripts/auditer_phase2_anti_fuite.py).

## Méthode

Le script lit les trois splits et contrôle :

- la structure des colonnes et la séparation de `parcours_cible` ;
- la présence exacte d'identifiants de parcours dans les features ;
- des motifs de données personnelles dans les noms de colonnes et les valeurs ;
- les doublons et les cibles inconnues ;
- les valeurs vides ;
- les distributions de `moyenne_scolaire` par classe ;
- les moyennes de compétences par classe ;
- une mesure de dépendance numérique (`eta squared`) ;
- une mesure exploratoire de dépendance catégorielle (NMI) ;
- la reconstruction de la cible par l'ancienne règle déterministe V1.

Les mesures d'association ne sont pas des performances ML et ne permettent pas de conclure à une capacité de généralisation.

## Contrôles directs

| Contrôle | Résultat |
|---|---:|
| Lignes auditées | 2 000 |
| Train / validation / test | 1 400 / 300 / 300 |
| Colonnes personnelles détectées | 0 |
| Motifs email/téléphone détectés | 0 |
| Identifiants de parcours dans les features hors cible | 0 |
| Doublons `id_candidat` | 0 |
| Cibles inconnues | 0 |
| Valeurs invalides détectées | 0 |
| Classes observées | 16 / 16 |
| Valeurs `projets` vides | 1 218 (60,90 %) |

Les colonnes V2 sont : `id_candidat`, `matieres_preferees`, `moyenne_scolaire`, `competences`, `centres_interet`, `projets`, `preferences_professionnelles`, `environnement_travail` et `parcours_cible`.

Le contrôle personnel est un contrôle de risque sur les colonnes et motifs courants. Il ne constitue pas une expertise juridique ni une preuve absolue sur l'origine des valeurs. Les données sont toutefois identifiées par le projet comme synthétiques et aucune donnée personnelle réelle n'a été détectée.

## Distributions par classe

### Cible

| Parcours | Nombre |
|---|---:|
| `parcours-esiia` | 185 |
| `parcours-imticia` | 160 |
| `parcours-emp` | 152 |
| `parcours-gca` | 144 |
| `parcours-isaia` | 140 |
| `parcours-tee` | 137 |
| `parcours-pip` | 129 |
| `parcours-dtja` | 127 |
| `parcours-caa` | 123 |
| `parcours-aee` | 115 |
| `parcours-emii` | 112 |
| `parcours-fic` | 105 |
| `parcours-teh` | 97 |
| `parcours-icmp` | 97 |
| `parcours-iaa` | 90 |
| `parcours-igglia` | 87 |

Le ratio classe majoritaire/classe minoritaire est de 2,13. La distribution n'est pas uniforme, mais aucune classe n'est absente.

### `moyenne_scolaire`

Les moyennes par classe vont de 12,446 (`parcours-gca`) à 13,872 (`parcours-pip`). L'eta squared entre la moyenne et la cible est de `0,0109`, ce qui indique une faible part de variance inter-classe dans cet audit descriptif. Cela n'exclut pas une relation exploitable par un modèle, mais ne montre pas une séparation déterministe.

### `competences`

Les quatre clés sont exactement celles des compétences documentées par le corpus. Les moyennes par classe restent globalement proches de 2,9 à 3,3 selon la compétence et la classe. Aucun profil ne contient de nouvelle compétence institutionnelle.

La V2 conserve néanmoins une dépendance conceptuelle aux compétences : elles participent faiblement à l'utilité synthétique de la cible lorsqu'un parcours possède une compétence reliée. Elles doivent donc faire l'objet d'une ablation en Phase 2.

### Variables catégorielles

- `projets` est vide dans 60,90 % des lignes ; cette absence est structurelle et synthétique.
- `preferences_professionnelles` et `environnement_travail` ont des modalités limitées et sont générés sans données personnelles.
- `centres_interet` et `matieres_preferees` sont synthétiques.
- `matieres_preferees` est une variable à forte cardinalité. Une NMI calculée sur les chaînes exactes peut être artificiellement élevée lorsque de nombreuses modalités sont rares ; cette mesure ne doit donc pas être interprétée comme une preuve de fuite.

La NMI exploratoire obtenue est : `matieres_preferees` 0,399 ; `centres_interet` 0,2582 ; `projets` 0,0062 ; `preferences_professionnelles` 0,0091 ; `environnement_travail` 0,0064. Ces chiffres servent de signal d'exploration, pas de critère d'acceptation isolé. Les variables à listes devront être transformées en indicateurs d'items ou en représentation textuelle dans une étape ultérieure, avec une nouvelle vérification de fuite.

## Comparaison V1/V2

| Version | Reconstruction par l'ancienne règle |
|---|---:|
| V1 | 100 % |
| V2 | 143 / 2 000 = 7,15 % |

En V1, `parcours_cible` était exactement le maximum d'une fonction utilisant `moyenne_scolaire` et `competences`. Il s'agissait d'une fuite de cible déterministe.

En V2, l'ancienne règle ne reconstruit que 143 observations. Les 7,15 % correspondent à des coïncidences entre l'ancienne règle et le nouveau tirage probabiliste. Ce taux ne mesure pas une fuite résiduelle et ne prouve pas que les features sont indépendantes de la cible.

La V2 utilise encore certaines variables observées dans son utilité latente, mais avec des poids faibles et un bruit de choix non exposé. La dépendance est donc synthétique et probabiliste, non une copie déterministe. Une dépendance de fond reste volontairement présente et devra être quantifiée avec les ablations prévues, sans entraîner de modèle pendant cet audit.

## Verdict de l'étape 1

**AUDIT ANTI-FUITE VALIDÉ AVEC RÉSERVES**

Aucune fuite directe ou reconstruction exacte par la règle V1 n'a été détectée dans la V2. La V2 peut passer à la préparation des features sous réserve de conserver les contrôles suivants :

1. `parcours_cible` ne doit jamais entrer dans les transformations de features ;
2. les statistiques d'encodage doivent être ajustées sur le train uniquement ;
3. les performances devront être comparées avec et sans moyenne et compétences ;
4. les variables synthétiques ne devront pas être interprétées comme des observations réelles ;
5. la forte cardinalité des listes devra être traitée sans créer de nouvelle fuite ;
6. le test final devra rester isolé jusqu'à la fin du protocole.

Cette validation ne signifie pas que le dataset est valide pour l'orientation réelle. Elle signifie uniquement que l'étape d'audit structurel et anti-fuite est suffisamment documentée pour autoriser la préparation des features.

## Validation technique

Les commandes demandées après cette étape sont :

- `pytest` ;
- `ruff check orient_ia scripts tests` ;
- `git status`.

`ruff check .` doit être contrôlé séparément, sans modifier `temp_run.py` si l'anomalie préexistante est toujours présente.
