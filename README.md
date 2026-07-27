# 🔍 Détecteur d'images générées par IA — Vision Transformer

> Un classifieur qui distingue une photographie réelle d'une image générée par IA — **et une investigation sur les raisons de ses échecs**.

**En bref** : 99 % d'accuracy sur le jeu de test interne, **79 % sur un corpus externe** constitué à la main. Cet écart de 23 points est le véritable objet de ce projet : le mesurer, le diagnostiquer par explicabilité, et tester expérimentalement les remèdes.

---

## 🎯 Le problème

Les générateurs d'images (Midjourney, Flux, SD3, DALL·E) produisent aujourd'hui des images photoréalistes indiscernables à l'œil nu. Détecter automatiquement ces contenus devient un enjeu concret : vérification journalistique, modération de plateformes, preuve judiciaire, protection du droit d'auteur.

La difficulté n'est pas d'entraîner un classifieur — c'est d'en entraîner un qui **généralise**. Un détecteur apprend les signatures des générateurs présents dans son corpus ; les générateurs évoluent plus vite que les datasets.

---

## 🏗️ Architecture

Fine-tuning d'un **Vision Transformer** pré-entraîné sur ImageNet, adapté à une classification binaire.

| Étape | Choix technique | Justification |
|---|---|---|
| **Corpus** | [`Parveshiiii/AI-vs-Real`](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) — 13 999 images HD | CIFAKE (32×32) est trop basse résolution pour les artefacts fins ; ArtiFact (31,7 Go) est inexploitable sur Colab |
| **Chargement** | `snapshot_download` + lecture parquet via pandas | Les métadonnées du dépôt sont incohérentes (9 999 annoncées vs 13 999 réelles), ce qui fait échouer `load_dataset` |
| **Rééquilibrage** | Sous-échantillonnage → 3 333 / 3 333 | Le corpus brut est à 76 % « réel » : un modèle non corrigé atteindrait 76 % d'accuracy en répondant toujours REAL |
| **Split** | 70 / 15 / 15, stratifié | `stratify` préserve le ratio 50/50 dans chaque sous-ensemble |
| **Modèle** | `google/vit-base-patch16-224` (86 M paramètres) | Transfer learning : le modèle « sait déjà voir », seule la tête de classification est neuve |
| **Entraînement** | 3 epochs, LR 2e-5, sélection sur F1 | LR faible pour ne pas détruire les poids pré-entraînés ; `load_best_model_at_end` protège du surapprentissage |

---

## 📊 Résultats

### En distribution — jeu de test interne (1 000 images)

| Modèle | Accuracy | Précision FAKE | Rappel FAKE | Précision REAL | Rappel REAL |
|---|---|---|---|---|---|
| ViT 224 | 98,0 % | 0,98 | 0,99 | 0,99 | 0,98 |
| ViT 384 | **99,2 %** | 0,99 | 0,99 | 0,99 | 0,99 |

**Détection du surapprentissage** — sur le ViT 224, la training loss chute à 0,0003 à l'époque 3 tandis que la validation loss remonte (0,0251 → 0,0264). Le modèle mémorise au lieu de généraliser ; `load_best_model_at_end` conserve donc les poids de l'époque 2.


### Hors distribution — corpus externe (29 images collectées à la main)

19 images issues de générateurs récents, 10 photographies personnelles.

| Modèle | Accuracy | Rappel FAKE | Rappel REAL |
|---|---|---|---|
| ViT 224 | 69 % | **53 %** (10/19) | 100 % (10/10) |
| ViT 384 | 85 % | **79 %** (15/19) | 100 % (10/10) |

**L'écart est de 23 points** entre test interne et corpus externe. Deux observations :

1. **Le biais est unidirectionnel.** Toutes les erreurs vont dans le même sens : des images générées classées REAL. Le rappel REAL reste à 100 % dans les deux configurations. Le modèle ne se trompe pas au hasard — il penche systématiquement vers « authentique ».

2. **Les erreurs ne sont pas des hésitations.** Une image générée typique est classée REAL avec **98,2 % de confiance**. Un ajustement du seuil de décision serait donc inopérant : il ne s'agit pas d'un défaut de calibration mais d'une lacune de couverture du corpus d'entraînement.

---

## 🔬 Diagnostic par explicabilité (Grad-CAM)

Pour comprendre *pourquoi* le modèle échoue, les cartes d'activation ont été calculées sur les cas d'erreur.

**Cas d'étude** — une image représentant une femme allongée dans un champ peuplé de **moutons roses**, classée REAL à 99,9 %.

Les zones d'activation se concentrent sur :
- le visage du sujet,
- le motif floral de la robe,
- la texture de la laine.

Les zones ignorées (activation minimale) :
- **les moutons roses** — l'anomalie la plus évidente de l'image,
- le ciel uniforme, les collines lisses,
- la composition générale, physiquement impossible.

> **Conclusion du diagnostic** : le modèle a appris un critère de **qualité photographique**, non de **plausibilité du contenu**. Sa question implicite est « ces textures sont-elles crédibles ? », pas « cette scène est-elle possible ? ». Cela explique son effondrement face aux générateurs récents, dont les textures sont irréprochables.

*(Implémentation : Grad-CAM adapté au ViT — `reshape_transform` repliant la séquence de 196 patches en grille 14×14, couche cible `vit.layers[-1].layernorm_before`. La LayerNorm finale, appliquée après agrégation, produit des cartes plates et inexploitables.)*

---

## 🧪 Expériences menées

### Expérience 1 — La résolution est-elle le facteur limitant ?

**Hypothèse** : le redimensionnement en 224×224 efface les artefacts haute fréquence des générateurs, ce qui expliquerait les échecs hors distribution.

**Protocole** : réentraînement identique en `vit-base-patch16-384` (2,9× plus de pixels conservés), évaluation sur le même corpus externe.

**Résultat** : rappel FAKE de 53 % → 79 % . Des images rattrapées, une perdue.

**Conclusion nuancée** : l'amélioration existe mais reste marginale, et **n'est pas statistiquement significative** — un test de McNemar sur 4 images discordantes donne p ≈ 0,63. Avec 19 images générées, on observe une tendance, on ne tranche pas. Surtout, 7 images générées sur 19 passent toujours : la résolution n'était pas la cause principale, ce qui corrobore le diagnostic Grad-CAM.

### Expérience 2 — Signature spectrale (piste abandonnée)

**Hypothèse** : les générateurs laissent une trace dans le spectre de Fourier, indépendante du contenu de l'image.

**Protocole** : profils radiaux du spectre, moyennés sur 200 images par classe, avec fenêtrage de Hann pour supprimer la fuite spectrale.

**Résultat** : un écart net apparaît sur le corpus d'entraînement — les images générées conservent davantage d'énergie haute fréquence. Mais cet écart n'a pas pu être validé sur le corpus externe, et un facteur confondant reste possible : une différence de pipeline de compression entre les deux classes du dataset produirait le même effet.

**Statut** : piste non concluante en l'état. Documentée pour ce qu'elle vaut — un résultat négatif rigoureux vaut mieux qu'une conclusion hâtive.

---

## ⚠️ Limites et pistes d'amélioration

| Limite constatée | Cause identifiée | Remède |
|---|---|---|
| **Rappel FAKE à 63 % hors distribution** | Le corpus d'entraînement ne contient pas de générateurs récents | Enrichissement par images Flux / SD3 / Midjourney v6 |
| **Critère de texture, pas de cohérence** (Grad-CAM) | Un classifieur binaire ne raisonne pas sur le contenu | Coupler à un modèle vision-langage évaluant la plausibilité de la scène |
| **Surapprentissage dès l'époque 2-3** | 6 666 images pour 86 M de paramètres | Augmentation de données, gel partiel des couches, arrêt anticipé |
| **Corpus d'évaluation externe restreint** (29 images) | Constitution manuelle | Étendre à 100+ images par classe pour des conclusions statistiquement fondées |
| **Robustesse aux traitements non mesurée** | Hors périmètre V1 | Évaluer sur images recompressées, redimensionnées, filtrées |

### Architecture envisagée pour une V2

Les trois approches explorées échouent sur des cas **disjoints**, ce qui plaide pour un système hybride :

| Approche | Détecte | Échoue sur |
|---|---|---|
| Texture (ViT actuel) | Artefacts des générateurs connus | Textures modernes irréprochables |
| Spectrale | Signature du procédé de fabrication | Images recompressées ou redimensionnées |
| Vision-langage | Incohérences sémantiques (moutons roses) | Contenus plausibles mais générés |

---

## 📁 Structure du dépôt

```
detection-image-ia/
├── Projet_detection_image_IA.ipynb   # notebook complet et commenté
├── README.md
├── requirements.txt
└── resultats/
    ├── matrice_confusion.png
    ├── gradcam_moutons_roses.png     # la figure du diagnostic
    └── comparaison_224_384.md
```

Le notebook est organisé en deux parcours : **exécution complète** (sections 1-9) et **reprise de session** (section 0) qui recharge les artefacts persistés sur Drive.


## 🛠️ Stack technique

**Modèle** : PyTorch, HuggingFace Transformers (ViT) · **Données** : pandas, datasets, PIL · **Évaluation** : scikit-learn, seaborn · **Explicabilité** : pytorch-grad-cam · **Analyse spectrale** : NumPy (FFT)

---

## 💡 Ce que ce projet démontre

Au-delà du classifieur, la démarche : mesurer honnêtement la généralisation plutôt que se satisfaire d'un score interne, diagnostiquer les échecs par l'explicabilité plutôt que les constater, formuler une hypothèse et la tester expérimentalement, et reconnaître les limites statistiques d'un résultat obtenu sur un petit échantillon.

Un modèle à 99 % qui échoue à 79 % sur des données réelles n'est pas un bon modèle — c'est un point de départ correctement mesuré.
---

*Projet personnel · [LinkedIn](https://linkedin.com/in/konan-gervais-n-guessan) · [GitHub](https://github.com/Gervais-59)*
