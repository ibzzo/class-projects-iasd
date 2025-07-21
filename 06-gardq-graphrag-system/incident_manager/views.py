# incident_manager/views.py
from django.views.generic import View, TemplateView
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .kg_knowledge_system import KGIncidentSystem
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IncidentSubmissionView(TemplateView):
    """Vue principale pour la soumission des nouveaux incidents."""
    template_name = 'incident_submission.html'

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        """Ajoute les données de contexte pour le formulaire."""
        context = super().get_context_data(**kwargs)
        context.update({
            'impact_levels': ['High', 'Medium', 'Low'],
            'categories': ['Technical', 'Access', 'Security', 'Performance']
        })
        return context

class IncidentAnalysisView(View):
    """Vue pour l'analyse des incidents et la suggestion de solutions via le Knowledge Graph."""

    def post(self, request):
        """Traite la soumission d'un nouvel incident et retourne les suggestions."""
        try:
            # Récupération des données du formulaire
            data = json.loads(request.body)
            description = data.get('description', '')
            summary = data.get('summary', '')
            impact = data.get('impact', '')
            category = data.get('category', '')

            if not description:
                return JsonResponse({
                    'status': 'error',
                    'message': 'La description est requise'
                }, status=400)

            # Construction de la requête
            query = f"{summary} {description}"
            if impact:
                query += f" avec priorité {impact}"
            if category:
                query += f" dans la catégorie {category}"

            # Initialisation du système de connaissances KG
            kg_system = KGIncidentSystem()
            
            try:
                # Obtention des réponses via le système KG
                answer_result = kg_system.get_answer(query)
                
                # Format de réponse optimisé avec contenu structuré
                response_data = {
                    'status': 'success',
                    'suggestions': {
                        'recommended_solution': answer_result.get('answer', ''),
                        'structured_content': answer_result.get('structured_content', {'sections': []}),
                        'confidence_score': round(answer_result.get('confidence', 0) * 100, 2),
                        'answer_type': answer_result.get('answer_type', 'SOLUTION'),
                        'ticket_id': answer_result.get('ticket_id', ''),
                        'similar_cases': [
                            {
                                'ticket_id': case.get('ticket_id', ''),
                                'summary': case.get('summary', ''),
                                'solution': case.get('solution', ''),
                                'priority': case.get('priority', ''),
                                'similarity': round(case.get('confidence', 0) * 100, 2)
                            }
                            for case in answer_result.get('relevant_tickets', [])
                        ]
                    }
                }

                return JsonResponse(response_data)

            finally:
                kg_system.close()

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de l'incident: {str(e)}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'message': 'Une erreur est survenue lors de l\'analyse'
            }, status=500)

class DataImportView(View):
    """Vue pour l'import des données historiques."""

    def post(self, request):
        try:
            file = request.FILES.get('file')
            if not file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Aucun fichier fourni'
                }, status=400)

            # Lecture et traitement du fichier Excel
            if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                try:
                    # Traitement du fichier Excel avec pandas
                    df = pd.read_excel(file)
                    
                    kg_system = KGIncidentSystem()
                    try:
                        # Import des données dans le système KG
                        kg_system.load_historical_data(df)
                        
                        # Récupération des statistiques pour confirmation
                        stats = kg_system.get_system_stats()
                        
                        return JsonResponse({
                            'status': 'success',
                            'message': f'{len(df)} incidents ont été importés avec succès',
                            'stats': stats
                        })
                    finally:
                        kg_system.close()
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du fichier Excel: {str(e)}", exc_info=True)
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Erreur lors du traitement du fichier: {str(e)}'
                    }, status=400)
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Format de fichier non supporté. Veuillez fournir un fichier Excel (.xlsx ou .xls)'
                }, status=400)

        except Exception as e:
            logger.error(f"Erreur lors de l'import des données: {str(e)}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'message': 'Une erreur est survenue lors de l\'import'
            }, status=500)

# Mise à jour de la classe MetricsView dans views.py
class MetricsView(View):
    """Vue pour les métriques de performance du système."""
    template_name = 'metrics.html'  # Ajout du template

    def get(self, request):
        # Déterminer si c'est une requête API ou une requête page web
        is_api_request = 'api' in request.path
        
        try:
            kg_system = KGIncidentSystem()
            try:
                # Récupération des statistiques du système KG
                stats = kg_system.get_system_stats()
                
                # Préparation des métriques pour l'affichage
                metrics = {
                    'total_tickets': stats.get('total_tickets', 0),
                    'total_sections': stats.get('total_sections', 0),
                    'intra_ticket_relations': stats.get('intra_ticket_relations', 0),
                    'explicit_relations': stats.get('explicit_inter_ticket_relations', 0),
                    'implicit_relations': stats.get('implicit_inter_ticket_relations', 0),
                    'section_types': stats.get('section_types', {})
                }
                
                if is_api_request:
                    # Pour les requêtes API, retourner JSON
                    return JsonResponse({
                        'status': 'success',
                        'metrics': metrics
                    })
                else:
                    # Pour les requêtes web, afficher le template
                    return render(request, self.template_name, {
                        'metrics': metrics
                    })
            finally:
                kg_system.close()

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques: {str(e)}", exc_info=True)
            
            if is_api_request:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Erreur lors de la récupération des métriques'
                }, status=500)
            else:
                return render(request, self.template_name, {
                    'error': 'Une erreur est survenue lors de la récupération des métriques.'
                })

class KGVisualizationView(TemplateView):
    """Vue pour la visualisation du graphe de connaissances."""
    template_name = 'kg_visualization.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class KGDataView(View):
    """Vue pour récupérer les données du graphe à visualiser."""
    
    # Remplacer cette méthode dans la classe KGDataView dans views.py

    def get(self, request):
        try:
            # Récupérer les paramètres
            ticket_id = request.GET.get('ticket_id', None)
            depth = int(request.GET.get('depth', 2))
            
            kg_system = KGIncidentSystem()
            try:
                with kg_system.driver.session() as session:
                    # Requête neo4j pour récupérer le sous-graphe
                    if ticket_id:
                        # Récupération d'un sous-graphe centré sur un ticket spécifique avec la profondeur demandée
                        query = f"""
                        MATCH path = (t:Ticket {{ticket_id: $ticket_id}})-[*1..{depth}]-(related)
                        RETURN path, relationships(path) as rels
                        LIMIT 200
                        """
                        result = session.run(query, {"ticket_id": ticket_id})
                    else:
                        # Récupération d'un échantillon général du graphe
                        query = f"""
                        MATCH (t:Ticket)
                        WITH t ORDER BY t.ticket_id LIMIT 5
                        MATCH path = (t)-[*1..{depth}]-(related)
                        RETURN path, relationships(path) as rels
                        LIMIT 200
                        """
                        result = session.run(query)
                    
                    # Préparation des données pour D3.js
                    nodes_dict = {}
                    links = []
                    
                    # Traitement des résultats
                    for record in result:
                        path = record["path"]
                        
                        # Pour chaque nœud dans le chemin
                        nodes_in_path = path.nodes
                        for node in nodes_in_path:
                            # Éviter les doublons en vérifiant si le nœud existe déjà
                            if node.id not in nodes_dict:
                                # Déterminer le type du nœud (première étiquette)
                                node_labels = list(node.labels)
                                node_type = node_labels[0] if node_labels else "Unknown"
                                
                                # Extraire toutes les propriétés
                                properties = dict(node)
                                
                                # Déterminer l'étiquette affichée en fonction du type de nœud
                                if node_type == "Ticket":
                                    label = f"Ticket {properties.get('ticket_id', 'N/A')}"
                                elif node_type == "Section":
                                    section_type = properties.get('type', '')
                                    label = f"{section_type}: {properties.get('content', '')[:30]}..."
                                elif node_type in ["Priority", "Cause", "Application", "Element"]:
                                    label = properties.get('name', 'N/A')
                                else:
                                    label = node_type
                                
                                # Créer l'objet nœud complet
                                nodes_dict[node.id] = {
                                    "id": node.id,
                                    "type": node_type,
                                    "label": label,
                                    "properties": properties
                                }
                        
                        # Pour chaque relation dans le chemin
                        rels = record["rels"]
                        for rel in rels:
                            # Créer l'objet lien
                            link_data = {
                                "id": rel.id,
                                "source": rel.start_node.id,
                                "target": rel.end_node.id,
                                "type": rel.type
                            }
                            
                            # Ajouter les propriétés de la relation si elles existent
                            rel_props = dict(rel)
                            if rel_props:
                                link_data["properties"] = rel_props
                            
                            links.append(link_data)
                    
                    # Conversion en liste pour le format final
                    nodes_list = list(nodes_dict.values())
                    
                    # Résultat final structuré pour D3.js
                    return JsonResponse({
                        "status": "success",
                        "nodes": nodes_list,
                        "links": links
                    })
            finally:
                kg_system.close()
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données du graphe: {str(e)}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'message': f'Erreur lors de la récupération des données du graphe: {str(e)}'
            }, status=500)