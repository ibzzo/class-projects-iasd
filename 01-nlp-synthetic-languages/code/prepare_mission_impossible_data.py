#!/usr/bin/env python3
"""
Script complet pour préparer les données de l'expérience "Mission: Impossible Language Models"
Version française avec données FQuAD

Auteur: Assistant Claude
Date: 2024
"""

import json
import random
import os
import re
import sys
from typing import List, Dict, Any
from collections import Counter

# ============================================================================
# SECTION 1: INSTALLATION DES DÉPENDANCES
# ============================================================================

print("=" * 80)
print("MISSION: IMPOSSIBLE LANGUAGE MODELS - PRÉPARATION DES DONNÉES")
print("=" * 80)

# Vérifier et installer les dépendances
print("\n📦 Vérification des dépendances...")

try:
    from datasets import load_dataset
    print("✓ datasets installé")
except ImportError:
    print("Installation de datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

try:
    import spacy
    print("✓ spaCy installé")
except ImportError:
    print("Installation de spaCy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy

# Télécharger le modèle français si nécessaire
try:
    nlp = spacy.load("fr_core_news_sm")
    print("✓ Modèle français spaCy chargé")
except OSError:
    print("Téléchargement du modèle français...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load("fr_core_news_sm")

# Configuration
random.seed(42)

# Créer la structure de dossiers
os.makedirs('synthetic_languages/shuffle', exist_ok=True)
os.makedirs('synthetic_languages/reverse', exist_ok=True)
os.makedirs('synthetic_languages/hop', exist_ok=True)

# ============================================================================
# SECTION 2: EXTRACTION DES DONNÉES DEPUIS FQUAD
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 1: EXTRACTION DES DONNÉES")
print("=" * 80)

# Charger le dataset FQuAD
print("\n📚 Chargement du dataset FQuAD...")
ds = load_dataset("manu/fquad2_test")

# Extraire tous les contextes uniques
contexts = set()
for split_name in ds.keys():
    print(f"  - Traitement du split '{split_name}': {len(ds[split_name])} exemples")
    for item in ds[split_name]:
        if 'context' in item:
            contexts.add(item['context'])

contexts_list = list(contexts)
print(f"\n✓ {len(contexts_list)} contextes uniques extraits")

# ============================================================================
# SECTION 3: DIVISION DES CONTEXTES EN PHRASES
# ============================================================================

def split_into_sentences(text: str) -> List[str]:
    """
    Divise un texte en phrases en gérant les abréviations françaises
    Filtre les phrases entre 10 et 50 mots
    """
    # Gérer les abréviations courantes pour éviter les fausses coupures
    abbreviations = {
        'M.': 'M@', 'Mme.': 'Mme@', 'Dr.': 'Dr@', 'Prof.': 'Prof@',
        'etc.': 'etc@', 'cf.': 'cf@', 'p.': 'p@', 'ex.': 'ex@',
        'c.-à-d.': 'c@-à-d@', 'av.': 'av@', 'apr.': 'apr@', 'J.-C.': 'J@-C@',
    }
    
    # Remplacer temporairement les abréviations
    for abbr, replacement in abbreviations.items():
        text = text.replace(abbr, replacement)
    
    # Diviser par les ponctuations de fin de phrase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ÿ])', text)
    
    # Restaurer les abréviations
    for abbr, replacement in abbreviations.items():
        sentences = [s.replace(replacement, abbr) for s in sentences]
    
    # Nettoyer et filtrer par longueur
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if 10 <= len(s.split()) <= 50]
    
    return sentences

print("\n📝 Division des contextes en phrases...")
all_sentences = []
for i, context in enumerate(contexts_list):
    if i % 100 == 0:
        print(f"  Progression: {i}/{len(contexts_list)}")
    sentences = split_into_sentences(context)
    all_sentences.extend(sentences)

# Dédupliquer les phrases
unique_sentences = list(set(all_sentences))
print(f"\n✓ {len(unique_sentences)} phrases uniques extraites")

# Créer les splits train/validation/test
random.shuffle(unique_sentences)
total = len(unique_sentences)
train_size = int(0.8 * total)  # 80%
val_size = int(0.1 * total)    # 10%

train_sentences = unique_sentences[:train_size]
val_sentences = unique_sentences[train_size:train_size + val_size]
test_sentences = unique_sentences[train_size + val_size:]

print(f"\n📊 Splits créés:")
print(f"  - Train: {len(train_sentences)} phrases")
print(f"  - Validation: {len(val_sentences)} phrases")
print(f"  - Test: {len(test_sentences)} phrases")

# ============================================================================
# SECTION 4: ANNOTATION POS AVEC SPACY
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 2: ANNOTATION POS")
print("=" * 80)

def annotate_sentence(sentence: str, nlp) -> Dict:
    """
    Annote une phrase avec les informations POS (Part-of-Speech)
    Nécessaire pour les transformations *Hop
    """
    doc = nlp(sentence)
    
    tokens = []
    for token in doc:
        token_info = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,        # POS universel
            "tag": token.tag_,        # POS détaillé
            "dep": token.dep_,        # Dépendance syntaxique
            "morph": str(token.morph), # Traits morphologiques
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop,
            "is_punct": token.is_punct
        }
        
        # Informations spécifiques pour les verbes (pour *Hop)
        if token.pos_ == "VERB":
            morph_dict = token.morph.to_dict()
            token_info["verb_form"] = morph_dict.get("VerbForm", "")
            token_info["tense"] = morph_dict.get("Tense", "")
            token_info["person"] = morph_dict.get("Person", "")
            token_info["number"] = morph_dict.get("Number", "")
        
        tokens.append(token_info)
    
    return {
        "original": sentence,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "num_words": len([t for t in tokens if t["is_alpha"]]),
        "has_verb": any(t["pos"] == "VERB" for t in tokens)
    }

print("\n🏷️  Annotation des phrases avec spaCy...")
tagged_data = {
    "metadata": {
        "source": "FQuAD contexts",
        "total_unique_sentences": len(unique_sentences),
        "language": "fr",
        "tagging_model": "fr_core_news_sm",
        "description": "Phrases annotées pour l'expérience Mission: Impossible Language Models"
    },
    "train": [],
    "validation": [],
    "test": []
}

# Annoter le train set
print("\n  Annotation du train set...")
for i, sentence in enumerate(train_sentences):
    if i % 200 == 0:
        print(f"    Progression: {i}/{len(train_sentences)}")
    tagged_data["train"].append(annotate_sentence(sentence, nlp))

# Annoter le validation set
print("\n  Annotation du validation set...")
for i, sentence in enumerate(val_sentences):
    if i % 50 == 0:
        print(f"    Progression: {i}/{len(val_sentences)}")
    tagged_data["validation"].append(annotate_sentence(sentence, nlp))

# Annoter le test set
print("\n  Annotation du test set...")
for i, sentence in enumerate(test_sentences):
    if i % 50 == 0:
        print(f"    Progression: {i}/{len(test_sentences)}")
    tagged_data["test"].append(annotate_sentence(sentence, nlp))

# Sauvegarder les données annotées
with open('tagged_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(tagged_data, f, ensure_ascii=False, indent=2)

print("\n✓ Données annotées sauvegardées dans 'tagged_sentences.json'")

# Statistiques sur les POS
verb_count = sum(1 for sent in tagged_data["train"] if sent["has_verb"])
print(f"\n📊 Statistiques:")
print(f"  - Phrases avec verbes dans le train: {verb_count}/{len(tagged_data['train'])} ({verb_count/len(tagged_data['train'])*100:.1f}%)")

# ============================================================================
# SECTION 5: DÉFINITION DES TRANSFORMATIONS
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 3: CRÉATION DES LANGUES SYNTHÉTIQUES")
print("=" * 80)

# --------------- Transformations *SHUFFLE ---------------

def no_shuffle(sentence_data: Dict) -> str:
    """NoShuffle: Retourne la phrase originale (contrôle)"""
    return sentence_data['original']

def nondeterministic_shuffle(sentence_data: Dict) -> str:
    """NondeterministicShuffle: Mélange aléatoire complet des mots"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    random.shuffle(tokens)
    return ' '.join(tokens)

def deterministic_shuffle(sentence_data: Dict) -> str:
    """DeterministicShuffle: Mélange déterministe basé sur la longueur"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    # Utiliser la longueur comme seed pour un mélange reproductible
    rng = random.Random(len(tokens))
    shuffled = tokens.copy()
    rng.shuffle(shuffled)
    return ' '.join(shuffled)

def local_shuffle(sentence_data: Dict) -> str:
    """LocalShuffle: Mélange par petits blocs de 2-3 mots"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    result = []
    i = 0
    
    while i < len(tokens):
        # Taille du bloc aléatoire (2 ou 3 mots)
        block_size = random.choice([2, 3])
        block = tokens[i:i+block_size]
        # Mélanger le bloc
        random.shuffle(block)
        result.extend(block)
        i += block_size
    
    return ' '.join(result)

def even_odd_shuffle(sentence_data: Dict) -> str:
    """EvenOddShuffle: Séparer positions paires/impaires puis concaténer"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    even_tokens = [tokens[i] for i in range(0, len(tokens), 2)]
    odd_tokens = [tokens[i] for i in range(1, len(tokens), 2)]
    return ' '.join(even_tokens + odd_tokens)

# --------------- Transformations *REVERSE ---------------

def no_reverse(sentence_data: Dict) -> str:
    """NoReverse: Retourne la phrase originale (contrôle)"""
    return sentence_data['original']

def partial_reverse(sentence_data: Dict) -> str:
    """PartialReverse: Inverser une partie après un marqueur R placé aléatoirement"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    
    if len(tokens) < 3:
        return ' '.join(tokens)
    
    # Placer le marqueur R à une position aléatoire
    r_position = random.randint(1, len(tokens) - 2)
    
    # Construire: début + R + fin inversée
    result = tokens[:r_position] + ['R'] + tokens[r_position:][::-1]
    return ' '.join(result)

def full_reverse(sentence_data: Dict) -> str:
    """FullReverse: Inverser complètement l'ordre des mots"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    return ' '.join(tokens[::-1])

# --------------- Transformations *HOP ---------------

def get_verb_indices(sentence_data: Dict) -> List[int]:
    """Trouve tous les indices des verbes dans la phrase"""
    return [i for i, token in enumerate(sentence_data['tokens']) 
            if token['pos'] == 'VERB']

def inflect_verb(token: Dict, should_pluralize: bool) -> str:
    """
    Inflexion simple du verbe français
    Note: Implémentation basique, une vraie conjugaison nécessiterait plus de règles
    """
    text = token['text']
    
    # Si on doit mettre au pluriel un verbe singulier
    if should_pluralize and token.get('number') == 'Sing':
        if text.endswith('e'):
            return text + 'nt'
        elif text.endswith('t'):
            return text[:-1] + 'ent'
        else:
            return text + 'ent'
    # Si on doit mettre au singulier un verbe pluriel
    elif not should_pluralize and token.get('number') == 'Plur':
        if text.endswith('ent'):
            return text[:-3]
        elif text.endswith('nt'):
            return text[:-2]
    
    return text

def no_hop(sentence_data: Dict) -> str:
    """NoHop: Ajouter des marqueurs S/P mais sans transformation du verbe"""
    tokens = [t['text'] for t in sentence_data['tokens']]
    verb_indices = get_verb_indices(sentence_data)
    
    if not verb_indices:
        return ' '.join(tokens)
    
    # Choisir un verbe aléatoire
    verb_idx = random.choice(verb_indices)
    
    # Placer un marqueur S (singulier) ou P (pluriel)
    marker = random.choice(['S', 'P'])
    marker_position = max(0, verb_idx - random.randint(1, min(4, verb_idx)) if verb_idx > 0 else 0)
    
    result = tokens[:marker_position] + [marker] + tokens[marker_position:]
    return ' '.join(result)

def token_hop(sentence_data: Dict) -> str:
    """TokenHop: Inflexion du verbe basée sur le comptage de tokens après le marqueur"""
    tokens_data = sentence_data['tokens']
    tokens = [t['text'] for t in tokens_data]
    verb_indices = get_verb_indices(sentence_data)
    
    if not verb_indices:
        return ' '.join(tokens)
    
    verb_idx = random.choice(verb_indices)
    marker = random.choice(['S', 'P'])
    marker_position = max(0, verb_idx - random.randint(5, min(8, verb_idx)) if verb_idx > 5 else 0)
    
    # Compter 4 tokens après le marqueur
    count_position = marker_position + 4
    
    # Règle impossible: inflexion basée sur la parité de la position
    should_pluralize = (count_position % 2 == 0) if marker == 'S' else (count_position % 2 == 1)
    
    # Construire la phrase avec verbe modifié
    result = tokens[:marker_position] + [marker] + tokens[marker_position:verb_idx]
    result.append(inflect_verb(tokens_data[verb_idx], should_pluralize))
    result.extend(tokens[verb_idx + 1:])
    
    return ' '.join(result)

def word_hop(sentence_data: Dict) -> str:
    """WordHop: Inflexion du verbe basée sur le comptage de mots (non-ponctuation)"""
    tokens_data = sentence_data['tokens']
    tokens = [t['text'] for t in tokens_data]
    verb_indices = get_verb_indices(sentence_data)
    
    if not verb_indices:
        return ' '.join(tokens)
    
    verb_idx = random.choice(verb_indices)
    marker = random.choice(['S', 'P'])
    marker_position = max(0, verb_idx - random.randint(5, min(8, verb_idx)) if verb_idx > 5 else 0)
    
    # Compter 4 mots (pas la ponctuation) après le marqueur
    word_count = 0
    for i in range(marker_position, len(tokens_data)):
        if tokens_data[i]['is_alpha']:
            word_count += 1
            if word_count == 4:
                break
    
    # Règle impossible: inflexion basée sur si on a atteint 4 mots
    should_pluralize = (word_count == 4) if marker == 'S' else (word_count != 4)
    
    # Construire la phrase
    result = tokens[:marker_position] + [marker] + tokens[marker_position:verb_idx]
    result.append(inflect_verb(tokens_data[verb_idx], should_pluralize))
    result.extend(tokens[verb_idx + 1:])
    
    return ' '.join(result)

# ============================================================================
# SECTION 6: APPLICATION DES TRANSFORMATIONS
# ============================================================================

# Dictionnaire de toutes les transformations
transformations = {
    # *Shuffle (5 variantes)
    'NoShuffle': no_shuffle,
    'NondeterministicShuffle': nondeterministic_shuffle,
    'DeterministicShuffle': deterministic_shuffle,
    'LocalShuffle': local_shuffle,
    'EvenOddShuffle': even_odd_shuffle,
    # *Reverse (3 variantes)
    'NoReverse': no_reverse,
    'PartialReverse': partial_reverse,
    'FullReverse': full_reverse,
    # *Hop (3 variantes)
    'NoHop': no_hop,
    'TokenHop': token_hop,
    'WordHop': word_hop
}

print("\n🔄 Application des 11 transformations...")

# Appliquer chaque transformation
for trans_name, trans_func in transformations.items():
    print(f"\n  Transformation: {trans_name}")
    
    # Créer le dataset transformé
    transformed_data = {
        "metadata": {
            **tagged_data["metadata"],
            "transformation": trans_name,
            "transformation_type": "shuffle" if "Shuffle" in trans_name else "reverse" if "Reverse" in trans_name else "hop"
        }
    }
    
    # Transformer chaque split
    for split in ["train", "validation", "test"]:
        transformed_data[split] = []
        
        # Réinitialiser la seed pour la reproductibilité
        random.seed(42)
        
        errors = 0
        for sentence_data in tagged_data[split]:
            try:
                transformed_text = trans_func(sentence_data)
                transformed_data[split].append({
                    "original": sentence_data["original"],
                    "transformed": transformed_text,
                    "num_tokens": sentence_data["num_tokens"],
                    "has_verb": sentence_data["has_verb"]
                })
            except Exception as e:
                # En cas d'erreur, utiliser l'original
                errors += 1
                transformed_data[split].append({
                    "original": sentence_data["original"],
                    "transformed": sentence_data["original"],
                    "num_tokens": sentence_data["num_tokens"],
                    "has_verb": sentence_data["has_verb"]
                })
        
        if errors > 0:
            print(f"    ⚠️  {errors} erreurs dans le split {split}")
    
    # Déterminer le sous-dossier
    if trans_name.endswith("Shuffle"):
        subdir = "shuffle"
    elif trans_name.endswith("Reverse"):
        subdir = "reverse"
    else:
        subdir = "hop"
    
    # Sauvegarder
    output_path = f"synthetic_languages/{subdir}/{trans_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)
    
    print(f"    ✓ Sauvegardé dans {output_path}")
    
    # Afficher un exemple
    example = transformed_data["train"][0]
    print(f"    Exemple: {example['original'][:50]}...")
    print(f"    → {example['transformed'][:50]}...")

# ============================================================================
# SECTION 7: RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)

print("\n✅ Préparation des données terminée avec succès!")
print(f"\n📊 Statistiques finales:")
print(f"  - Contextes FQuAD traités: {len(contexts_list)}")
print(f"  - Phrases uniques extraites: {len(unique_sentences)}")
print(f"  - Répartition: {len(train_sentences)} train / {len(val_sentences)} val / {len(test_sentences)} test")
print(f"  - Transformations appliquées: 11")

print("\n📁 Fichiers créés:")
print("  - tagged_sentences.json (données annotées)")
print("  - synthetic_languages/")
print("    ├── shuffle/ (5 fichiers)")
print("    ├── reverse/ (3 fichiers)")
print("    └── hop/ (3 fichiers)")

print("\n🎯 Prochaine étape:")
print("  Utiliser ces données pour entraîner des modèles de langage")
print("  et comparer les performances entre langues naturelles et impossibles")

print("\n" + "=" * 80)
print("FIN DU SCRIPT")
print("=" * 80)