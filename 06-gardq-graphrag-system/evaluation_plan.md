# Plan d'Évaluation Expérimentale pour GARDQ

## 📊 Vue d'ensemble

Ce document présente un plan d'évaluation complet pour le système GARDQ, basé sur la méthodologie du papier LinkedIn (arxiv:2404.17723) et les meilleures pratiques actuelles en évaluation des systèmes RAG.

## 🎯 Objectifs de l'Évaluation

1. **Évaluer la performance de récupération** : Mesurer la capacité du système à retrouver les tickets pertinents
2. **Évaluer la qualité de génération** : Vérifier la pertinence et la qualité des solutions suggérées
3. **Mesurer l'impact opérationnel** : Quantifier la réduction du temps de résolution
4. **Valider l'approche Knowledge Graph** : Comparer avec des approches baseline

## 📐 Métriques d'Évaluation

### 1. Métriques de Récupération (Retrieval)

#### a) **MRR (Mean Reciprocal Rank)**
- Mesure la position moyenne du premier résultat pertinent
- Formule : MRR = (1/|Q|) × Σ(1/rank_i)
- Cible : > 0.7 (amélioration de 77.6% comme LinkedIn)

#### b) **Recall@K** (K = 1, 3, 5, 10)
- Proportion de tickets pertinents dans les K premiers résultats
- Formule : Recall@K = |relevant ∩ retrieved_K| / |relevant|
- Cibles :
  - Recall@1 : > 0.5
  - Recall@5 : > 0.8
  - Recall@10 : > 0.9

#### c) **NDCG@K** (Normalized Discounted Cumulative Gain)
- Évalue l'ordre des résultats en tenant compte de leur pertinence
- Particulièrement important pour les suggestions ordonnées

### 2. Métriques de Génération

#### a) **BLEU Score**
- Compare les solutions générées avec des solutions de référence
- Cible : Amélioration de 0.32+ par rapport au baseline

#### b) **ROUGE-L**
- Mesure la plus longue sous-séquence commune
- Utile pour évaluer la cohérence structurelle

#### c) **METEOR**
- Prend en compte les synonymes et paraphrases
- Plus adapté aux réponses techniques variées

#### d) **BERTScore**
- Évaluation sémantique basée sur les embeddings
- Capture mieux le sens que les métriques basées sur les n-grammes

### 3. Métriques Spécifiques au Domaine

#### a) **Taux de Résolution Exacte**
- % de cas où la solution suggérée correspond exactement à la solution appliquée
- Formule : solutions_exactes / total_tickets

#### b) **Couverture des Entités**
- % d'entités du ticket correctement identifiées (Application, Priority, etc.)
- Mesure la compréhension du contexte

#### c) **Temps de Résolution Médian**
- Réduction du temps entre soumission et résolution
- Cible : -28.6% comme LinkedIn

#### d) **Taux d'Hallucination**
- % de suggestions contenant des informations non vérifiables
- Cible : < 10%

## 🧪 Setup Expérimental

### 1. Préparation des Données

#### Dataset Principal
```python
# Structure du dataset d'évaluation
{
    "test_set": {
        "size": 1000,  # tickets pour l'évaluation
        "split": "80/10/10",  # train/val/test
        "stratification": "par type d'incident et priorité"
    },
    "annotations": {
        "tickets_similaires": "3-5 par ticket test",
        "solutions_reference": "solutions validées par experts",
        "entites_annotees": "toutes les entités métier"
    }
}
```

#### Création du Ground Truth
1. Sélectionner 1000 tickets résolus avec succès
2. Pour chaque ticket :
   - Identifier manuellement 3-5 tickets similaires
   - Valider la solution appliquée
   - Annoter les entités métier

### 2. Baselines de Comparaison

#### a) **BM25 Baseline**
- Recherche textuelle classique sans graphe
- Implémentation avec Elasticsearch ou Lucene

#### b) **Embeddings Purs**
- Recherche vectorielle seule (sans graphe)
- Utiliser le même modèle d'embeddings

#### c) **RAG Simple**
- Récupération + génération sans structure de graphe
- GPT-4 avec contexte de tickets concaténés

#### d) **Version simplifiée**
- Version sans extraction de sous-graphes
- Recherche par embeddings uniquement

### 3. Protocole d'Évaluation

#### Phase 1 : Évaluation de la Récupération
```python
def evaluate_retrieval(system, test_queries, ground_truth):
    metrics = {
        'mrr': [],
        'recall@1': [],
        'recall@5': [],
        'recall@10': [],
        'ndcg@5': []
    }
    
    for query in test_queries:
        results = system.search(query, k=10)
        relevant = ground_truth[query.id]
        
        # Calculer les métriques
        metrics['mrr'].append(calculate_mrr(results, relevant))
        metrics['recall@1'].append(calculate_recall_at_k(results, relevant, k=1))
        # ...
        
    return aggregate_metrics(metrics)
```

#### Phase 2 : Évaluation de la Génération
```python
def evaluate_generation(system, test_tickets, reference_solutions):
    metrics = {
        'bleu': [],
        'rouge': [],
        'meteor': [],
        'bertscore': []
    }
    
    for ticket in test_tickets:
        generated = system.generate_solution(ticket)
        reference = reference_solutions[ticket.id]
        
        # Calculer les métriques
        metrics['bleu'].append(calculate_bleu(generated, reference))
        # ...
        
    return aggregate_metrics(metrics)
```

#### Phase 3 : Évaluation End-to-End
```python
def evaluate_end_to_end(system, test_scenarios):
    results = {
        'resolution_accuracy': [],
        'entity_coverage': [],
        'hallucination_rate': [],
        'time_to_solution': []
    }
    
    for scenario in test_scenarios:
        start_time = time.time()
        suggestion = system.process_incident(scenario.query)
        
        # Évaluer la qualité complète
        results['resolution_accuracy'].append(
            check_resolution_match(suggestion, scenario.ground_truth)
        )
        # ...
        
    return aggregate_results(results)
```

## 📊 Plan d'Implémentation

### Semaine 1-2 : Préparation des Données
- [ ] Sélectionner et nettoyer 1000 tickets
- [ ] Créer les annotations manuelles
- [ ] Développer les scripts de prétraitement

### Semaine 3 : Implémentation des Baselines
- [ ] BM25 avec Elasticsearch
- [ ] Système d'embeddings purs
- [ ] RAG simple sans graphe

### Semaine 4 : Framework d'Évaluation
- [ ] Implémenter les métriques de récupération
- [ ] Implémenter les métriques de génération
- [ ] Créer le pipeline d'évaluation automatisé

### Semaine 5-6 : Expérimentations
- [ ] Évaluer tous les systèmes
- [ ] Ablation studies sur GARDQ
- [ ] Analyse des erreurs

### Semaine 7 : Analyse et Rapport
- [ ] Analyse statistique des résultats
- [ ] Génération des graphiques
- [ ] Rédaction du rapport d'évaluation

## 🔬 Ablation Studies

### 1. Impact du Knowledge Graph
- Comparer GARDQ avec/sans structure de graphe
- Mesurer l'apport de chaque type de relation

### 2. Taille du Contexte
- Varier k (nombre de tickets récupérés)
- Identifier le trade-off optimal

### 3. Modèles de Génération
- Comparer GPT-4, GPT-3.5, Claude
- Évaluer l'impact sur la qualité/coût

### 4. Types d'Embeddings
- Tester différents modèles (multilingual, domain-specific)
- Mesurer l'impact sur la récupération

## 📈 Résultats Attendus

### Amélioration Cibles (vs Baseline BM25)
- MRR : +70-80%
- BLEU : +0.3-0.4
- Temps de résolution : -25-30%
- Taux d'hallucination : < 10%

### Livrables
1. **Rapport technique** : Méthodologie complète et résultats
2. **Dashboard de métriques** : Visualisation interactive
3. **Code d'évaluation** : Framework réutilisable
4. **Dataset annoté** : Pour recherches futures

## 🚀 Prochaines Étapes

1. **Validation du plan** avec les parties prenantes
2. **Collecte des données** d'évaluation
3. **Développement** du framework d'évaluation
4. **Exécution** des expériences
5. **Publication** des résultats

Ce plan d'évaluation permet une validation rigoureuse du système GARDQ tout en fournissant des insights actionnables pour son amélioration continue.