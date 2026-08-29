# Phase 1 - Corpus pedagogique

## Perimetre

Le corpus version `2026.08.26` contient uniquement les informations visibles le
26 aout 2026 sur les pages institutionnelles suivantes :

- `http://www.ispm-edu.com/presentation.php` ;
- `http://www.ispm-edu.com/filieres.php` ;
- `http://www.ispm-edu.com/inscription.php`.

La fiche `https://annuaire.mg/ispm/` est conservee comme source externe de
corroboration et de contradiction. Elle ne fournit aucune donnee d'entite
retenue comme fait institutionnel.

## Donnees integrees

- 6 mentions et 16 parcours annonces par la presentation de l'ISPM ;
- 2 niveaux/diplomes annonces : Licence BACC+3 et Master BACC+5 ;
- 4 descriptions de competences generales presentes sur la page des filieres ;
- 5 prerequis ou conditions d'acces explicitement publies ;
- 1 debouche ISAIA explicitement publie : banques, entreprises industrielles
  et entreprises commerciales.

Chaque entite porte un ou plusieurs identifiants de source. Chaque source
conserve son titre, origine, URL, date de consultation, statut, section, extrait,
donnees extraites, limites et incertitudes. Le chargement du corpus refuse une
entite sans provenance ou une reference de source inconnue.

## Informations volontairement absentes

Le corpus ne contient aucune matiere principale de formation : les matieres
trouvees dans la source externe sont des matieres de concours et ne sont pas
transformees en maquette pedagogique.

Il ne contient pas non plus de referentiel exhaustif de competences, de metiers
parcours par parcours, ni de passerelle. Aucune passerelle n'est deduite d'une
simple proximite entre intitules ou niveaux.

L'affirmation d'habilitation ministerielle presente sur le site de l'ISPM n'a pas
ete verifiee par un acte officiel independant. Elle reste donc une affirmation
institutionnelle avec cette incertitude explicitement conservee.

## Contradictions conservees

Les contradictions sont stockees dans `contradictions` dans le fichier JSON :

- organisation de Genie Industriel et Genie Civil ;
- selection de dossier sur le site ISPM contre concours dans l'annuaire ;
- Licence/Master sur le site ISPM contre Ingenieur/Doctorat dans l'annuaire.

Aucune version contradictoire n'est fusionnee arbitrairement.

## Hors perimetre

Cette etape ne cree pas de donnees synthetiques, de profil ML, d'entrainement,
de RAG ou d'agent. Ces elements restent des etapes ulterieures.
