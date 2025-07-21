# Mission: Impossible Language Models - Version Française

Reproduction de l'expérience 1 de l'article ["Mission: Impossible Language Models"](https://arxiv.org/abs/2401.06416v2) avec des données en français.

## 🎯 Objectif

Tester si les modèles de langage apprennent aussi facilement des "langues impossibles" (avec des structures grammaticales non-naturelles) que des langues naturelles, en utilisant des données françaises du dataset FQuAD.

## 📋 Structure du projet

```
.
├── prepare_mission_impossible_data.py  # Script principal de préparation des données
├── plan_experience_impossible_languages_fr.md  # Plan détaillé de l'expérience
├── 2401.06416v2.pdf  # Article de référence
├── README.md  # Ce fichier
└── venv/  # Environnement virtuel Python
```

### Fichiers générés après exécution :

```
├── tagged_sentences.json  # 2,814 phrases annotées avec POS tags
└── synthetic_languages/   # 11 transformations
    ├── shuffle/          # 5 variantes
    ├── reverse/          # 3 variantes
    └── hop/              # 3 variantes
```

## 🚀 Installation et utilisation

### 1. Cloner le repository

```bash
git clone [URL_DU_REPO]
cd mission-impossible-languages-fr
```

### 2. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python3 -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

### 3. Exécuter le script de préparation

```bash
python prepare_mission_impossible_data.py
```

Le script :
- Installe automatiquement les dépendances (`datasets`, `spacy`)
- Télécharge le modèle français de spaCy
- Extrait ~2,814 phrases uniques depuis FQuAD
- Applique 11 transformations syntaxiques

**Temps d'exécution** : 5-10 minutes

## 📊 Dataset

- **Source** : FQuAD (`manu/fquad2_test`) via Hugging Face
- **Nombre de phrases** : 2,814 uniques
- **Longueur** : 10-50 mots par phrase
- **Splits** : 80% train / 10% validation / 10% test
- **Langue** : Français

## 🔄 Transformations (Table 1 de l'article)

### Famille *Shuffle (5 variantes)
1. **NoShuffle** : Contrôle (phrases originales)
2. **NondeterministicShuffle** : Mélange aléatoire
3. **DeterministicShuffle** : Mélange basé sur la longueur
4. **LocalShuffle** : Mélange par blocs de 2-3 mots
5. **EvenOddShuffle** : Séparation positions paires/impaires

### Famille *Reverse (3 variantes)
1. **NoReverse** : Contrôle (phrases originales)
2. **PartialReverse** : Inversion partielle avec marqueur R
3. **FullReverse** : Inversion complète

### Famille *Hop (3 variantes)
1. **NoHop** : Marqueurs S/P sans transformation
2. **TokenHop** : Inflexion verbale basée sur comptage de tokens
3. **WordHop** : Inflexion verbale basée sur comptage de mots

## 📈 Prochaines étapes

1. **Entraînement** : Créer `train_models.py` pour entraîner des modèles sur chaque transformation
2. **Évaluation** : Mesurer la perplexité et la vitesse de convergence
3. **Analyse** : Comparer les performances entre langues naturelles et impossibles

## 📚 Références

- Article original : [Mission: Impossible Language Models](https://arxiv.org/abs/2401.06416v2)
- Dataset : [FQuAD sur Hugging Face](https://huggingface.co/datasets/manu/fquad2_test)
- Modèle spaCy : [fr_core_news_sm](https://spacy.io/models/fr)

## 📝 License

Ce projet est à des fins de recherche académique uniquement.

## 👤 Auteur

dauphinoi

---

*Note : Ce projet est une adaptation en français de l'expérience originale qui utilisait des données en anglais.*