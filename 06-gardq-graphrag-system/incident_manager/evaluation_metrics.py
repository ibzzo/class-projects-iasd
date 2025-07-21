"""
Module d'évaluation pour le système GARDQ
Implémente les métriques du papier LinkedIn SIGIR '24
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd
from sklearn.metrics import ndcg_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch
from datetime import datetime
import json


class GARDQEvaluator:
    """
    Classe principale pour l'évaluation du système GARDQ
    Implémente toutes les métriques du papier LinkedIn
    """
    
    def __init__(self, kg_system=None):
        self.kg_system = kg_system
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
        
    # ================ MÉTRIQUES DE RÉCUPÉRATION ================
    
    def calculate_mrr(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Calcule le Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_ids: Liste ordonnée des IDs récupérés
            relevant_ids: Ensemble des IDs pertinents
            
        Returns:
            MRR score (0-1)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_recall_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calcule le Recall@K
        
        Args:
            retrieved_ids: Liste ordonnée des IDs récupérés
            relevant_ids: Ensemble des IDs pertinents
            k: Nombre de résultats à considérer
            
        Returns:
            Recall@K score (0-1)
        """
        retrieved_k = set(retrieved_ids[:k])
        if len(relevant_ids) == 0:
            return 0.0
        return len(retrieved_k.intersection(relevant_ids)) / len(relevant_ids)
    
    def calculate_ndcg_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], 
                           relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calcule le NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved_ids: Liste ordonnée des IDs récupérés
            relevant_ids: Ensemble des IDs pertinents
            relevance_scores: Scores de pertinence pour chaque document
            k: Nombre de résultats à considérer
            
        Returns:
            NDCG@K score (0-1)
        """
        # Scores réels pour les K premiers résultats
        actual_scores = []
        for doc_id in retrieved_ids[:k]:
            if doc_id in relevance_scores:
                actual_scores.append(relevance_scores[doc_id])
            else:
                actual_scores.append(0.0)
        
        # Scores idéaux (triés par pertinence)
        all_scores = sorted(relevance_scores.values(), reverse=True)
        ideal_scores = all_scores[:k]
        
        # Pad avec des zéros si nécessaire
        while len(actual_scores) < k:
            actual_scores.append(0.0)
        while len(ideal_scores) < k:
            ideal_scores.append(0.0)
            
        # Calculer NDCG
        actual_scores = np.array(actual_scores).reshape(1, -1)
        ideal_scores = np.array(ideal_scores).reshape(1, -1)
        
        return ndcg_score(ideal_scores, actual_scores)
    
    # ================ MÉTRIQUES DE GÉNÉRATION ================
    
    def calculate_bleu(self, generated: str, references: List[str]) -> Dict[str, float]:
        """
        Calcule les scores BLEU (1-4)
        
        Args:
            generated: Texte généré
            references: Liste de textes de référence
            
        Returns:
            Dict avec BLEU-1 à BLEU-4
        """
        # Tokenisation simple
        generated_tokens = generated.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]
        
        scores = {}
        for n in range(1, 5):
            weights = [1/n] * n + [0] * (4-n)
            score = sentence_bleu(
                reference_tokens, 
                generated_tokens,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            scores[f'bleu_{n}'] = score
            
        return scores
    
    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calcule les scores ROUGE
        
        Args:
            generated: Texte généré
            reference: Texte de référence
            
        Returns:
            Dict avec ROUGE-1, ROUGE-2, ROUGE-L
        """
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge_1': scores['rouge1'].fmeasure,
            'rouge_2': scores['rouge2'].fmeasure,
            'rouge_l': scores['rougeL'].fmeasure
        }
    
    def calculate_bert_score(self, generated: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calcule le BERTScore
        
        Args:
            generated: Liste de textes générés
            references: Liste de textes de référence
            
        Returns:
            Dict avec precision, recall, f1
        """
        P, R, F1 = bert_score(generated, references, lang='fr', device='cpu')
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    
    # ================ MÉTRIQUES SPÉCIFIQUES AU DOMAINE ================
    
    def calculate_entity_coverage(self, extracted_entities: Set[str], 
                                 ground_truth_entities: Set[str]) -> float:
        """
        Calcule la couverture des entités extraites
        
        Args:
            extracted_entities: Entités extraites par le système
            ground_truth_entities: Entités de référence
            
        Returns:
            Score de couverture (0-1)
        """
        if not ground_truth_entities:
            return 1.0 if not extracted_entities else 0.0
            
        return len(extracted_entities.intersection(ground_truth_entities)) / len(ground_truth_entities)
    
    def calculate_hallucination_rate(self, generated_solution: str, 
                                   source_tickets: List[Dict]) -> float:
        """
        Calcule le taux d'hallucination
        
        Args:
            generated_solution: Solution générée
            source_tickets: Tickets sources utilisés
            
        Returns:
            Taux d'hallucination (0-1)
        """
        # Extraire toutes les informations factuelles des tickets sources
        source_facts = set()
        for ticket in source_tickets:
            # Ajouter les faits du ticket (simplification)
            if 'description' in ticket:
                source_facts.update(ticket['description'].lower().split())
            if 'solution' in ticket:
                source_facts.update(ticket['solution'].lower().split())
        
        # Vérifier les faits dans la solution générée
        generated_facts = set(generated_solution.lower().split())
        
        # Calculer les faits non supportés (approximation simple)
        unsupported = generated_facts - source_facts
        
        if not generated_facts:
            return 0.0
            
        return len(unsupported) / len(generated_facts)
    
    def calculate_resolution_accuracy(self, suggested_solution: str, 
                                    actual_solution: str, 
                                    threshold: float = 0.8) -> bool:
        """
        Vérifie si la solution suggérée correspond à la solution réelle
        
        Args:
            suggested_solution: Solution suggérée par le système
            actual_solution: Solution réellement appliquée
            threshold: Seuil de similarité pour considérer une correspondance
            
        Returns:
            True si les solutions correspondent
        """
        # Utiliser ROUGE-L pour la similarité
        rouge_score = self.calculate_rouge(suggested_solution, actual_solution)
        return rouge_score['rouge_l'] >= threshold
    
    # ================ ÉVALUATION COMPLÈTE ================
    
    def evaluate_retrieval(self, test_queries: List[Dict], 
                          ground_truth: Dict[str, Dict]) -> Dict[str, float]:
        """
        Évalue la performance de récupération sur un ensemble de test
        
        Args:
            test_queries: Liste des requêtes de test
            ground_truth: Vérité terrain pour chaque requête
            
        Returns:
            Dict avec toutes les métriques de récupération
        """
        metrics = defaultdict(list)
        
        for query in test_queries:
            query_id = query['id']
            query_text = query['description']
            
            # Récupérer les résultats
            retrieved = self.kg_system.search_similar_tickets(query_text, k=10)
            retrieved_ids = [r['ticket_id'] for r in retrieved]
            
            # Obtenir la vérité terrain
            relevant_ids = set(ground_truth[query_id]['relevant_tickets'])
            relevance_scores = ground_truth[query_id].get('relevance_scores', {})
            
            # Calculer les métriques
            metrics['mrr'].append(self.calculate_mrr(retrieved_ids, relevant_ids))
            metrics['recall@1'].append(self.calculate_recall_at_k(retrieved_ids, relevant_ids, 1))
            metrics['recall@5'].append(self.calculate_recall_at_k(retrieved_ids, relevant_ids, 5))
            metrics['recall@10'].append(self.calculate_recall_at_k(retrieved_ids, relevant_ids, 10))
            
            if relevance_scores:
                metrics['ndcg@5'].append(
                    self.calculate_ndcg_at_k(retrieved_ids, relevant_ids, relevance_scores, 5)
                )
                metrics['ndcg@10'].append(
                    self.calculate_ndcg_at_k(retrieved_ids, relevant_ids, relevance_scores, 10)
                )
        
        # Moyenner les métriques
        return {metric: np.mean(values) for metric, values in metrics.items()}
    
    def evaluate_generation(self, test_cases: List[Dict]) -> Dict[str, float]:
        """
        Évalue la qualité de génération sur des cas de test
        
        Args:
            test_cases: Liste des cas de test avec solutions de référence
            
        Returns:
            Dict avec toutes les métriques de génération
        """
        all_generated = []
        all_references = []
        metrics = defaultdict(list)
        
        for case in test_cases:
            query = case['query']
            reference_solution = case['reference_solution']
            
            # Générer la solution
            result = self.kg_system.analyze_incident(query)
            generated_solution = result['suggested_solution']
            
            all_generated.append(generated_solution)
            all_references.append(reference_solution)
            
            # Métriques individuelles
            bleu_scores = self.calculate_bleu(generated_solution, [reference_solution])
            rouge_scores = self.calculate_rouge(generated_solution, reference_solution)
            
            for key, value in bleu_scores.items():
                metrics[key].append(value)
            for key, value in rouge_scores.items():
                metrics[key].append(value)
        
        # BERTScore sur l'ensemble
        bert_scores = self.calculate_bert_score(all_generated, all_references)
        
        # Moyenner et combiner
        averaged_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        averaged_metrics.update(bert_scores)
        
        return averaged_metrics
    
    def run_full_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """
        Exécute une évaluation complète du système
        
        Args:
            test_data_path: Chemin vers les données de test
            
        Returns:
            Rapport d'évaluation complet
        """
        # Charger les données de test
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'system': 'GARDQ',
            'metrics': {}
        }
        
        # Évaluation de la récupération
        if 'retrieval_test' in test_data:
            retrieval_metrics = self.evaluate_retrieval(
                test_data['retrieval_test']['queries'],
                test_data['retrieval_test']['ground_truth']
            )
            results['metrics']['retrieval'] = retrieval_metrics
        
        # Évaluation de la génération
        if 'generation_test' in test_data:
            generation_metrics = self.evaluate_generation(
                test_data['generation_test']['cases']
            )
            results['metrics']['generation'] = generation_metrics
        
        # Calculer les améliorations par rapport au baseline
        if 'baseline_results' in test_data:
            results['improvements'] = self._calculate_improvements(
                results['metrics'],
                test_data['baseline_results']
            )
        
        return results
    
    def _calculate_improvements(self, current_metrics: Dict, 
                               baseline_metrics: Dict) -> Dict[str, float]:
        """
        Calcule les améliorations par rapport au baseline
        """
        improvements = {}
        
        for category in current_metrics:
            if category in baseline_metrics:
                for metric, current_value in current_metrics[category].items():
                    if metric in baseline_metrics[category]:
                        baseline_value = baseline_metrics[category][metric]
                        if baseline_value > 0:
                            improvement = ((current_value - baseline_value) / baseline_value) * 100
                            improvements[f"{category}_{metric}"] = improvement
        
        return improvements


# ================ UTILITAIRES D'ÉVALUATION ================

def create_evaluation_report(results: Dict[str, Any], output_path: str):
    """
    Crée un rapport d'évaluation formaté
    """
    report = f"""
# Rapport d'Évaluation GARDQ
Date: {results['timestamp']}

## Métriques de Récupération
"""
    
    if 'retrieval' in results['metrics']:
        for metric, value in results['metrics']['retrieval'].items():
            report += f"- {metric}: {value:.4f}\n"
    
    report += "\n## Métriques de Génération\n"
    
    if 'generation' in results['metrics']:
        for metric, value in results['metrics']['generation'].items():
            report += f"- {metric}: {value:.4f}\n"
    
    if 'improvements' in results:
        report += "\n## Améliorations vs Baseline\n"
        for metric, improvement in results['improvements'].items():
            report += f"- {metric}: {improvement:+.1f}%\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


def prepare_test_data(tickets_df: pd.DataFrame, num_samples: int = 1000) -> Dict:
    """
    Prépare les données de test à partir d'un DataFrame de tickets
    """
    # Échantillonner les tickets
    test_tickets = tickets_df.sample(n=min(num_samples, len(tickets_df)))
    
    test_data = {
        'retrieval_test': {
            'queries': [],
            'ground_truth': {}
        },
        'generation_test': {
            'cases': []
        }
    }
    
    # Créer les cas de test
    for idx, ticket in test_tickets.iterrows():
        query_id = f"test_{idx}"
        
        # Test de récupération
        test_data['retrieval_test']['queries'].append({
            'id': query_id,
            'description': ticket['Short description']
        })
        
        # Pour la vérité terrain, on pourrait utiliser des tickets similaires
        # (à implémenter selon votre logique métier)
        
        # Test de génération
        if 'Resolution notes' in ticket and pd.notna(ticket['Resolution notes']):
            test_data['generation_test']['cases'].append({
                'query': ticket['Short description'],
                'reference_solution': ticket['Resolution notes']
            })
    
    return test_data