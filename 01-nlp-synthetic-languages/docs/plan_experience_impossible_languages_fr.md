# Plan pour reproduire l'expérience "Mission: Impossible Language Models" en français

## Contexte
Reproduction de l'expérience 1 de l'étude "Mission: Impossible Language Models" (arXiv:2401.06416v2) qui teste si les modèles de langage apprennent aussi facilement des langues "impossibles" que des langues naturelles.

## Dataset de base
- **2,814 phrases uniques en français** extraites du dataset FQuAD (`manu/fquad2_test`)
- Phrases de longueur entre 10 et 50 mots (moyenne: 25.5 mots)
- Splits : 2,251 train / 281 validation / 282 test
- Générées automatiquement par le script

## Étapes de l'expérience

### 1. Préparation des données
- **Script unique**: `prepare_mission_impossible_data.py`
- **Fonctionnalités**:
  - Installation automatique des dépendances (datasets, spaCy)
  - Extraction des contextes depuis FQuAD
  - Division en phrases (10-50 mots)
  - Annotation POS avec spaCy français
  - Application des 11 transformations
- **Exécution**: `python prepare_mission_impossible_data.py`
- **Temps estimé**: 5-10 minutes

### 2. Création des langues synthétiques impossibles (Table 1 de l'article)

#### Famille *Shuffle (5 variantes)
1. **NoShuffle**: Phrases originales sans modification (contrôle)
2. **NondeterministicShuffle**: Mélange aléatoire des mots
   - Exemple: "Le chat mange la souris" → "souris mange le chat la"
3. **DeterministicShuffle**: Mélange déterministe basé sur la longueur de la phrase
   - Utilise une seed = longueur de la phrase pour le mélange
4. **LocalShuffle**: Mélange par blocs locaux de 2-3 mots
   - Exemple: "Le chat noir mange la souris" → "chat Le noir la mange souris"
5. **EvenOddShuffle**: Séparer mots pairs/impairs puis concaténer
   - Exemple: "Le chat mange la souris" → "Le mange souris chat la"

#### Famille *Reverse (3 variantes)
1. **NoReverse**: Phrases originales (contrôle)
2. **PartialReverse**: Inverser une partie de la phrase après un marqueur 'R'
   - Exemple: "Le chat R mange la souris" → "Le chat souris la mange"
3. **FullReverse**: Inverser toute la phrase
   - Exemple: "Le chat mange la souris" → "souris la mange chat Le"

#### Famille *Hop (3 variantes) - Nécessite POS tagging
1. **NoHop**: Phrases avec marqueurs S/P mais sans transformation
2. **TokenHop**: Inflexion du verbe basée sur le comptage de tokens
   - Compte 4 tokens après le marqueur S/P pour déterminer l'inflexion
3. **WordHop**: Inflexion du verbe basée sur le comptage de mots
   - Compte 4 mots après le marqueur S/P pour déterminer l'inflexion

### 3. Entraînement des modèles

#### Configuration adaptée au dataset réduit
- **Modèle suggéré**: 
  - DistilGPT-2 ou GPT-2 nano (plus adapté à 2,814 phrases)
  - Ou CamemBERT-base / DistilCamemBERT pour le français
- **Hyperparamètres**:
  - Learning rate: 5e-5
  - Batch size: 8 (réduit pour le petit dataset)
  - Epochs: 50-100 (augmenté pour compenser le dataset réduit)
  - Early stopping basé sur la validation
  - Tokenizer: Adapté au français

#### Processus
1. Créer 11 datasets transformés (NoShuffle, 4 Shuffle, NoReverse, 2 Reverse, NoHop, 2 Hop)
2. Entraîner un modèle par transformation (11 modèles au total)
3. Sauvegarder des checkpoints tous les 5 epochs
4. Monitorer la perplexité sur validation à chaque epoch
5. Arrêter si pas d'amélioration pendant 10 epochs

### 4. Métriques d'évaluation

1. **Perplexité sur le test set**
   - Mesure principale de la qualité d'apprentissage
   - Comparer les courbes de perplexité entre langues

2. **Vitesse de convergence**
   - Nombre d'epochs pour atteindre une perplexité stable
   - Taux de diminution de la perplexité

3. **Analyse qualitative**
   - Génération de texte dans chaque langue
   - Évaluation de la cohérence des patterns appris

### 5. Analyses complémentaires

1. **Analyse des embeddings**
   - Comparer la structure des espaces d'embeddings
   - Mesurer la similarité cosinus entre mots

2. **Probing tasks**
   - Tester si le modèle a appris les règles implicites
   - Créer des tâches de classification simples

3. **Généralisation**
   - Tester sur des phrases avec des structures non vues
   - Évaluer la robustesse des patterns appris

## Structure du code

```
mission_impossible_languages_fr/
├── prepare_mission_impossible_data.py  # Script principal de préparation
├── plan_experience_impossible_languages_fr.md  # Ce plan
├── 2401.06416v2.pdf  # Article de référence
├── tagged_sentences.json  # Généré : phrases annotées POS
├── synthetic_languages/  # Généré : transformations
│   ├── shuffle/
│   │   ├── NoShuffle.json
│   │   ├── NondeterministicShuffle.json
│   │   ├── DeterministicShuffle.json
│   │   ├── LocalShuffle.json
│   │   └── EvenOddShuffle.json
│   ├── reverse/
│   │   ├── NoReverse.json
│   │   ├── PartialReverse.json
│   │   └── FullReverse.json
│   └── hop/
│       ├── NoHop.json
│       ├── TokenHop.json
│       └── WordHop.json
├── train_models.py  # À créer : entraînement
├── evaluate.py  # À créer : évaluation
├── models/  # À créer : checkpoints
├── results/  # À créer : résultats
└── README.md  # À créer : documentation
```

## Résultats attendus

Selon l'étude originale, on devrait observer:

### Hiérarchie de performance (du meilleur au pire):
1. **NoShuffle / NoReverse / NoHop** (contrôles) - Perplexité la plus basse
2. **LocalShuffle** - Perturbation locale, reste partiellement apprennable
3. **DeterministicShuffle / EvenOddShuffle** - Règles fixes mais non-naturelles
4. **PartialReverse** - Inversion partielle avec marqueur
5. **NondeterministicShuffle / FullReverse** - Destruction complète de la structure
6. **TokenHop / WordHop** - Règles de comptage impossibles

### Métriques spécifiques:
- Écart de perplexité: 2-10x entre contrôles et langues impossibles
- Vitesse de convergence: 2-3x plus lente pour langues impossibles
- Certaines langues impossibles peuvent ne jamais converger

## Notes importantes

- Utiliser une seed fixe (42) pour la reproductibilité
- Dataset réduit (2,814 vs 9.69M phrases) : résultats indicatifs mais tendances valides
- Documenter toutes les transformations appliquées avec exemples
- Sauvegarder les configurations d'entraînement en JSON
- GPU recommandé mais CPU possible avec le dataset réduit
- Temps estimé: 1-2h par modèle sur GPU, 4-8h sur CPU

## Limitations de l'adaptation

- Dataset 3,400x plus petit que l'original
- Modèles plus petits nécessaires
- Résultats moins robustes statistiquement
- Suffisant pour une preuve de concept et publication de workshop