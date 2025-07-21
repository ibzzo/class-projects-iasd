# incident_manager/kg_knowledge_system.py
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Chargement des variables d'environnement (clé API OpenAI)
load_dotenv()

logger = logging.getLogger(__name__)

class KGIncidentSystem:
    """
    Système de gestion d'incidents utilisant un Knowledge Graph avec LLM
    suivant exactement l'approche RAG avec Knowledge Graphs décrite dans le papier LinkedIn SIGIR '24
    """
    def __init__(self, uri: str = "neo4j://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "Ticket04#"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.logger = logging.getLogger(__name__)
        
        # Initialisation du client OpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY n'est pas définie dans les variables d'environnement")
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Configuration de la base de données
        self.setup_database()
        
        # Chargement du template pour la structure du graphe
        self.graph_template = self._load_graph_template()

    def _load_graph_template(self) -> Dict:
        """
        Charge le template YAML pour la structure du graphe.
        Dans une implémentation réelle, ceci serait chargé depuis un fichier.
        """
        return {
            "ticket": {
                "sections": [
                    {"type": "SUMMARY", "required": True},
                    {"type": "DESCRIPTION", "required": True},
                    {"type": "SOLUTION", "required": False},
                    {"type": "ROOT_CAUSE", "required": False},
                    {"type": "PRIORITY", "required": False},
                    {"type": "STATUS", "required": False},
                    {"type": "APPLICATION", "required": False},
                    {"type": "ELEMENT", "required": False},
                    {"type": "RESOLUTION_METHOD", "required": False},
                    {"type": "CREATION_DATE", "required": False},
                    {"type": "CLOSURE_DATE", "required": False}
                ],
                "relations": [
                    {"type": "PARENT_TICKET", "direction": "outgoing"},
                    {"type": "SIMILAR_TO", "direction": "bidirectional"}
                ]
            }
        }

    def setup_database(self):
        """Configure les contraintes et index dans Neo4j."""
        with self.driver.session() as session:
            try:
                # Contraintes pour unicité des nœuds
                constraints = [
                    "CREATE CONSTRAINT ticket_id IF NOT EXISTS FOR (t:Ticket) REQUIRE t.ticket_id IS UNIQUE",
                    # Utilisation d'une contrainte composite pour section_id et ticket_id
                    "CREATE CONSTRAINT section_composite_id IF NOT EXISTS FOR (s:Section) REQUIRE (s.section_id, s.ticket_id) IS UNIQUE",
                    "CREATE CONSTRAINT cause_name IF NOT EXISTS FOR (c:Cause) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT priority_name IF NOT EXISTS FOR (p:Priority) REQUIRE p.name IS UNIQUE",
                    "CREATE CONSTRAINT app_name IF NOT EXISTS FOR (a:Application) REQUIRE a.name IS UNIQUE",
                    "CREATE CONSTRAINT element_name IF NOT EXISTS FOR (e:Element) REQUIRE e.name IS UNIQUE",
                ]
                
                # Création des index pour améliorer les performances de recherche
                indexes = [
                    "CREATE INDEX ticket_summary IF NOT EXISTS FOR (t:Ticket) ON (t.summary)",
                    "CREATE INDEX section_type IF NOT EXISTS FOR (s:Section) ON (s.type)",
                    "CREATE INDEX section_embedding IF NOT EXISTS FOR (s:Section) ON (s.embedding)",
                    "CREATE INDEX section_ticket_id IF NOT EXISTS FOR (s:Section) ON (s.ticket_id)",
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Contrainte déjà existante ou erreur: {str(e)}")
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.warning(f"Index déjà existant ou erreur: {str(e)}")
                
                logger.info("Base de données configurée avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors de la configuration de la base : {str(e)}")
                raise

    def _llm_parse_ticket_to_tree(self, row: pd.Series) -> Dict:
        """
        Utilise le LLM pour analyser un ticket et le convertir en structure d'arbre
        selon la méthode décrite dans le papier LinkedIn.
        """
        ticket_id = str(row["numéro ticket"])
        
        # Conversion des données du ticket en texte pour le prompt
        ticket_text = "Données du ticket:\n"
        for col, value in row.items():
            if pd.notna(value) and value != "":
                ticket_text += f"{col}: {value}\n"
        
        # Construction du prompt pour le LLM
        template_str = json.dumps(self.graph_template, indent=2)
        prompt = f"""
        Analyse les données du ticket suivant et représente-le sous forme d'arbre structuré suivant le template fourni.
        Identifie clairement les différentes sections du ticket qui correspondent aux types définis dans le template.
        
        {ticket_text}
        
        Structure du template:
        {template_str}
        
        Retourne le résultat au format JSON structuré ainsi:
        {{
            "ticket_id": "ID_DU_TICKET",
            "sections": [
                {{
                    "section_id": "ID_UNIQUE_SECTION",
                    "type": "TYPE_DE_SECTION",
                    "content": "CONTENU_DE_LA_SECTION"
                }},
                ...
            ]
        }}
        
        N'inclus pas de sections avec des contenus vides.
        Retourne uniquement le JSON sans autre texte.
        """
        
        try:
            # Appel au LLM (GPT-4o-mini)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant expert en analyse de tickets d'incident et leur représentation en graphe de connaissances."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Température basse pour des réponses plus déterministes
                response_format={"type": "json_object"}
            )
            
            # Extraction et parsing du JSON
            llm_response = response.choices[0].message.content
            ticket_tree = json.loads(llm_response)
            
            # Validation minimale
            if "ticket_id" not in ticket_tree or "sections" not in ticket_tree:
                raise ValueError("Structure de réponse LLM invalide")
            
            # Post-traitement pour garantir l'unicité des section_id
            for section in ticket_tree["sections"]:
                # Génération d'un ID unique combinant ticket_id et type de section
                section["section_id"] = f"{ticket_tree['ticket_id']}_{section['type']}"
                
            return ticket_tree
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse LLM du ticket {ticket_id}: {str(e)}")
            # Fallback en cas d'erreur: utiliser une méthode basée sur des règles
            return self._rule_based_parse_ticket_to_tree(row)

    def _rule_based_parse_ticket_to_tree(self, row: pd.Series) -> Dict:
        """
        Méthode de fallback basée sur des règles en cas d'échec du LLM.
        """
        ticket_id = str(row["numéro ticket"])
        
        # Construction de l'arbre du ticket avec ses sections
        tree = {
            "ticket_id": ticket_id,
            "sections": []
        }
        
        # Mapping des colonnes aux types de sections
        sections_mapping = {
            "Résumé": "SUMMARY",
            "Description": "DESCRIPTION",
            "Solution Ticket": "SOLUTION",
            "Cause": "ROOT_CAUSE",
            "Priorité": "PRIORITY",
            "Etat ticket": "STATUS",
            "Application de Résolution": "APPLICATION",
            "Elément en Résolution": "ELEMENT",
            "Methode de resolution": "RESOLUTION_METHOD",
            "Date création ticket": "CREATION_DATE",
            "Date clôture ticket": "CLOSURE_DATE"
        }
        
        # Ajout des sections au ticket
        for col, section_type in sections_mapping.items():
            if col in row and pd.notna(row[col]) and row[col] != "":
                section_content = str(row[col])
                
                # Générer un ID unique pour la section combinant ticket_id et type
                section_id = f"{ticket_id}_{section_type}"
                
                # Ajouter la section à l'arbre
                tree["sections"].append({
                    "section_id": section_id,
                    "type": section_type,
                    "content": section_content
                })
        
        return tree

    def _create_intra_ticket_tree(self, session, tree: Dict):
        """
        Crée la structure d'arbre intra-ticket dans Neo4j.
        """
        # Création du nœud principal du ticket
        session.run("""}}
        MERGE (t:Ticket {ticket_id: $ticket_id})
        """, {"ticket_id": tree["ticket_id"]})
        
        # Création des nœuds de section et des relations avec le ticket
        for section in tree["sections"]:
            # Calcul d'embedding pour les sections textuelles importantes
            embedding = None
            if section["type"] in ["SUMMARY", "DESCRIPTION", "SOLUTION", "ROOT_CAUSE"]:
                embedding = self.encoder.encode(section["content"]).tolist()
            
            # Utiliser CREATE au lieu de MERGE pour éviter la réutilisation des sections
            query = """
            CREATE (s:Section {
                section_id: $section_id,
                type: $type, 
                content: $content,
                ticket_id: $ticket_id
            })
            """
            
            params = {
                "section_id": section["section_id"],
                "type": section["type"],
                "content": section["content"],
                "ticket_id": tree["ticket_id"]  # Ajout de l'ID du ticket comme propriété
            }
            
            # Ajout de l'embedding si disponible
            if embedding:
                query = query.replace("})", ", embedding: $embedding})")
                params["embedding"] = embedding
            
            session.run(query, params)
            
            # Relation entre le ticket et la section
            session.run("""
            MATCH (t:Ticket {ticket_id: $ticket_id})
            MATCH (s:Section {section_id: $section_id, ticket_id: $ticket_id})
            MERGE (t)-[r:HAS_SECTION]->(s)
            """, {
                "ticket_id": tree["ticket_id"],
                "section_id": section["section_id"]
            })
            
            # Création de nœuds spécifiques en fonction du type de section
            if section["type"] == "ROOT_CAUSE":
                session.run("""
                MERGE (c:Cause {name: $cause})
                WITH c
                MATCH (s:Section {section_id: $section_id, ticket_id: $ticket_id})
                MERGE (s)-[:REFERS_TO]->(c)
                """, {
                    "cause": section["content"],
                    "section_id": section["section_id"],
                    "ticket_id": tree["ticket_id"]
                })
            
            elif section["type"] == "PRIORITY":
                session.run("""
                MERGE (p:Priority {name: $priority})
                WITH p
                MATCH (s:Section {section_id: $section_id, ticket_id: $ticket_id})
                MERGE (s)-[:REFERS_TO]->(p)
                """, {
                    "priority": section["content"],
                    "section_id": section["section_id"],
                    "ticket_id": tree["ticket_id"]
                })
                
            elif section["type"] == "APPLICATION":
                session.run("""
                MERGE (a:Application {name: $app})
                WITH a
                MATCH (s:Section {section_id: $section_id, ticket_id: $ticket_id})
                MERGE (s)-[:REFERS_TO]->(a)
                """, {
                    "app": section["content"],
                    "section_id": section["section_id"],
                    "ticket_id": tree["ticket_id"]
                })
                
            elif section["type"] == "ELEMENT":
                session.run("""
                MERGE (e:Element {name: $element})
                WITH e
                MATCH (s:Section {section_id: $section_id, ticket_id: $ticket_id})
                MERGE (s)-[:REFERS_TO]->(e)
                """, {
                    "element": section["content"],
                    "section_id": section["section_id"],
                    "ticket_id": tree["ticket_id"]
                })

    def _create_inter_ticket_connections(self, session, df: pd.DataFrame):
        """
        Crée les connexions inter-tickets, à la fois explicites et implicites.
        """
        # 1. Connexions explicites (basées sur le numéro du ticket père)
        for idx, row in df.iterrows():
            ticket_id = str(row["numéro ticket"])
            parent_ticket_id = row.get("numéro du ticket père")
            
            if pd.notna(parent_ticket_id) and parent_ticket_id != "":
                parent_ticket_id = str(parent_ticket_id)
                session.run("""
                MATCH (t1:Ticket {ticket_id: $ticket_id})
                MATCH (t2:Ticket {ticket_id: $parent_ticket_id})
                MERGE (t1)-[r:PARENT_TICKET]->(t2)
                """, {
                    "ticket_id": ticket_id,
                    "parent_ticket_id": parent_ticket_id
                })
        
        # 2. Connexions implicites (basées sur la similarité sémantique)
        # Récupérer tous les tickets avec leur résumé
        result = session.run("""
        MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section)
        WHERE s.type = 'SUMMARY'
        RETURN t.ticket_id AS ticket_id, s.content AS summary, s.section_id AS section_id
        """)
        
        tickets = [(record["ticket_id"], record["summary"], record["section_id"]) for record in result]
        
        # Calculer les embeddings pour tous les résumés
        ticket_ids = [t[0] for t in tickets]
        summaries = [t[1] for t in tickets]
        section_ids = [t[2] for t in tickets]
        
        if not summaries:
            logger.warning("Aucun résumé trouvé pour établir des connexions implicites")
            return
            
        embeddings = self.encoder.encode(summaries)
        
        # Seuil de similarité minimal pour créer une connexion
        similarity_threshold = 0.8
        
        # Créer des connexions basées sur la similarité
        for i in range(len(ticket_ids)):
            for j in range(i+1, len(ticket_ids)):
                # Calcul de similarité cosinus
                similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                
                if similarity >= similarity_threshold:
                    session.run("""
                    MATCH (t1:Ticket {ticket_id: $ticket_id1})
                    MATCH (t2:Ticket {ticket_id: $ticket_id2})
                    MERGE (t1)-[r:SIMILAR_TO {similarity: $similarity}]->(t2)
                    """, {
                        "ticket_id1": ticket_ids[i],
                        "ticket_id2": ticket_ids[j],
                        "similarity": float(similarity)
                    })

    def load_historical_data(self, df: pd.DataFrame):
        """
        Charge les données des tickets dans le graphe de connaissances.
        """
        with self.driver.session() as session:
            # Traitement par lots pour éviter une surcharge de la base
            batch_size = 10
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(total_batches), desc="Batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                # Étape 1: Analyse et création des arbres intra-ticket
                for idx, row in batch_df.iterrows():
                    try:
                        # Utilisation du LLM pour transformer le ticket en arbre
                        tree = self._llm_parse_ticket_to_tree(row)
                        self._create_intra_ticket_tree(session, tree)
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement du ticket {row.get('numéro ticket', 'inconnu')}: {str(e)}")
                        continue
            
            # Étape 2: Création des connexions inter-tickets
            try:
                self._create_inter_ticket_connections(session, df)
                logger.info("Connexions inter-tickets créées avec succès")
            except Exception as e:
                logger.error(f"Erreur lors de la création des connexions inter-tickets: {str(e)}")

    def _llm_parse_query_entity_intent(self, query_text: str) -> Tuple[Dict[str, str], List[str]]:
        """
        Utilise un LLM pour analyser la requête et extraire les entités nommées et les intentions,
        conformément à la méthode décrite dans le papier.
        """
        # Construction du prompt pour le LLM
        template_str = json.dumps(self.graph_template, indent=2)
        prompt = f"""
        Analyse la requête utilisateur suivante et identifie les entités nommées et les intentions.
        
        Requête utilisateur: "{query_text}"
        
        Structure du graphe de tickets:
        {template_str}
        
        Retourne le résultat au format JSON structuré ainsi:
        {{
            "entities": {{
                "TYPE_ENTITE_1": "VALEUR_ENTITE_1",
                "TYPE_ENTITE_2": "VALEUR_ENTITE_2",
                ...
            }},
            "intents": ["INTENTION_1", "INTENTION_2", ...]
        }}
        
        Types d'entités possibles: SUMMARY, DESCRIPTION, SOLUTION, ROOT_CAUSE, PRIORITY, APPLICATION, ELEMENT, STATUS
        Intentions possibles: SOLUTION, ROOT_CAUSE, DESCRIPTION, STEPS_TO_REPRODUCE
        
        Extrais uniquement les entités et intentions qui sont clairement exprimées dans la requête.
        Retourne uniquement le JSON sans autre texte.
        """
        
        try:
            # Appel au LLM (GPT-4o-mini)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant expert en analyse de requêtes et extraction d'entités et d'intentions pour un système de graphe de connaissances."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Extraction et parsing du JSON
            llm_response = response.choices[0].message.content
            query_analysis = json.loads(llm_response)
            
            # Validation minimale
            if "entities" not in query_analysis or "intents" not in query_analysis:
                raise ValueError("Structure de réponse LLM invalide")
                
            return query_analysis["entities"], query_analysis["intents"]
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse LLM de la requête: {str(e)}")
            # Fallback: méthode basique de détection d'entités et d'intentions
            return self._basic_parse_query_entity_intent(query_text)

    def _basic_parse_query_entity_intent(self, query_text: str) -> Tuple[Dict[str, str], List[str]]:
        """
        Méthode de fallback basique pour l'analyse de requêtes en cas d'échec du LLM.
        """
        entities = {}
        intents = []
        
        # Détection basique d'entités et d'intentions
        query_lower = query_text.lower()
        
        # Détection d'entités
        if "priorité" in query_lower or "urgence" in query_lower:
            if "élevée" in query_lower or "haute" in query_lower or "urgent" in query_lower:
                entities["PRIORITY"] = "Élevée"
            elif "moyenne" in query_lower or "normale" in query_lower:
                entities["PRIORITY"] = "Normale"
            elif "basse" in query_lower or "faible" in query_lower:
                entities["PRIORITY"] = "Basse"
        
        if "application" in query_lower:
            parts = query_lower.split("application")
            if len(parts) > 1:
                app_part = parts[1].strip()
                app_words = app_part.split()
                if app_words:
                    entities["APPLICATION"] = app_words[0].capitalize()
        
        # Détection d'intentions
        if "solution" in query_lower or "résoudre" in query_lower or "comment" in query_lower:
            intents.append("SOLUTION")
        
        if "cause" in query_lower or "pourquoi" in query_lower:
            intents.append("ROOT_CAUSE")
            
        if "reproduire" in query_lower or "étapes" in query_lower or "comment faire" in query_lower:
            intents.append("STEPS_TO_REPRODUCE")
        
        if "description" in query_lower or "détail" in query_lower:
            intents.append("DESCRIPTION")
            
        # Si aucune intention spécifique n'est détectée, considérer qu'il s'agit d'une recherche de solution
        if not intents:
            intents.append("SOLUTION")
            
        # Extraction du résumé (mots-clés importants)
        summary_words = []
        for word in query_text.split():
            word = word.strip().lower()
            if len(word) > 3 and word not in ["comment", "pourquoi", "quand", "est-ce", "avec", "sans", "pour", "dans"]:
                summary_words.append(word)
        
        if summary_words:
            entities["SUMMARY"] = " ".join(summary_words)
            
        return entities, intents

    def _retrieve_relevant_tickets_embedding(self, session, entities: Dict[str, str], query_text: str, limit: int = 5) -> List[str]:
        """
        Récupère les tickets les plus pertinents en utilisant la recherche par embeddings,
        exactement comme décrit dans le papier LinkedIn.
        """
        try:
            # Construire une requête de recherche combinée à partir des entités et du texte original
            search_text = query_text
            
            # Préparation de l'embedding de recherche
            query_embedding = self.encoder.encode(search_text).tolist()
            
            # Vérifier si des embeddings existent dans la base de données
            check_query = """
            MATCH (s:Section)
            WHERE s.embedding IS NOT NULL
            RETURN COUNT(s) AS count
            """
            result = session.run(check_query)
            count = result.single()["count"]
            
            if count == 0:
                logger.warning("Aucun embedding trouvé dans la base de données, fallback vers une recherche textuelle")
                return self._retrieve_tickets_by_text(session, query_text, entities, limit)
            
            # Recherche par similarité vectorielle avec les embeddings
            # Cette requête est exactement comme décrite dans le papier, avec adaptation pour Neo4j
            vector_query = """
            MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section)
            WHERE s.embedding IS NOT NULL
            WITH t, s, gds.similarity.cosine(s.embedding, $query_embedding) AS similarity
            WHERE similarity > 0.5
            WITH t, MAX(similarity) AS max_similarity
            ORDER BY max_similarity DESC
            LIMIT $limit
            RETURN t.ticket_id AS ticket_id, max_similarity
            """
            
            # Exécution de la requête
            result = session.run(vector_query, {
                "query_embedding": query_embedding,
                "limit": limit
            })
            
            # Récupération des résultats
            tickets = []
            for record in result:
                tickets.append((record["ticket_id"], record["max_similarity"]))  # Tuple (ID, score)
            
            if tickets:
                logger.info(f"Tickets trouvés par recherche vectorielle: {tickets}")
                return tickets
            
            # Si aucun résultat, utiliser la recherche textuelle
            logger.warning("Aucun résultat avec la recherche vectorielle, fallback vers une recherche textuelle")
            return self._retrieve_tickets_by_text(session, query_text, entities, limit)
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche par embedding: {str(e)}")
            # Fallback vers la recherche textuelle
            return self._retrieve_tickets_by_text(session, query_text, entities, limit)
    
    def _retrieve_tickets_by_text(self, session, query_text: str, entities: Dict[str, str], limit: int = 5) -> List[Tuple[str, float]]:
        """
        Méthode de fallback pour rechercher des tickets par contenu textuel
        si la recherche par embedding échoue.
        """
        try:
            relevance_scores = {}
            
            # 1. Recherche dans les sections par type (SUMMARY, DESCRIPTION, etc.)
            section_types = ["SUMMARY", "DESCRIPTION", "SOLUTION", "ROOT_CAUSE"]
            
            for section_type in section_types:
                query = """
                MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section {type: $section_type})
                WHERE toLower(s.content) CONTAINS toLower($query_text)
                RETURN DISTINCT t.ticket_id AS ticket_id
                """
                
                result = session.run(query, {
                    "section_type": section_type,
                    "query_text": query_text.lower()
                })
                
                for record in result:
                    ticket_id = record["ticket_id"]
                    relevance_scores[ticket_id] = relevance_scores.get(ticket_id, 0) + 1
            
            # 2. Recherche par entités spécifiques
            for entity_type, entity_value in entities.items():
                if entity_type in ["PRIORITY", "APPLICATION", "ELEMENT"]:
                    query = f"""
                    MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section {{type: '{entity_type}'}})
                    WHERE toLower(s.content) CONTAINS toLower($entity_value)
                    RETURN DISTINCT t.ticket_id AS ticket_id
                    """
                    
                    result = session.run(query, {
                        "entity_value": entity_value.lower()
                    })
                    
                    for record in result:
                        ticket_id = record["ticket_id"]
                        relevance_scores[ticket_id] = relevance_scores.get(ticket_id, 0) + 2
            
            # 3. Recherche par mots-clés
            keywords = [word.strip().lower() for word in query_text.split() if len(word.strip()) > 3]
            
            for keyword in keywords:
                query = """
                MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section)
                WHERE toLower(s.content) CONTAINS $keyword
                RETURN DISTINCT t.ticket_id AS ticket_id
                """
                
                result = session.run(query, {"keyword": keyword})
                
                for record in result:
                    ticket_id = record["ticket_id"]
                    relevance_scores[ticket_id] = relevance_scores.get(ticket_id, 0) + 0.5
            
            # Trier les tickets par score de pertinence
            sorted_tickets = sorted(
                [(tid, score) for tid, score in relevance_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Retourner les top N tickets AVEC leurs scores (format: [(ticket_id, score), ...])
            return sorted_tickets[:limit]
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche textuelle: {str(e)}")
            return []

    def _llm_generate_graph_query(self, entities: Dict[str, str], intents: List[str], ticket_ids: List[str] = None) -> str:
        """
        Utilise un LLM pour générer des requêtes Cypher pour Neo4j
        en fonction des entités, intentions et tickets identifiés.
        """
        # Préparation des données pour le prompt
        entities_str = json.dumps(entities, indent=2)
        intents_str = json.dumps(intents, indent=2)
        tickets_str = json.dumps(ticket_ids if ticket_ids else [], indent=2)
        
        # Prompt beaucoup plus précis pour guider le LLM
        prompt = f"""
        Génère une requête Cypher pour Neo4j à partir des entités et intentions extraites d'une requête utilisateur.

        Entités identifiées:
        {entities_str}

        Intentions identifiées: 
        {intents_str}

        Tickets pertinents (si déjà identifiés):
        {tickets_str}

        Structure exacte du graphe:
        - (Ticket) n'a pas directement de propriétés de contenu, seulement ticket_id
        - (Ticket)-[:HAS_SECTION]->(Section {{type: "TYPE", content: "CONTENT", ticket_id: "TICKET_ID"}})
        - Les types de Section incluent: "SUMMARY", "DESCRIPTION", "SOLUTION", "ROOT_CAUSE", "PRIORITY", etc.
        - Pour accéder au contenu d'un ticket, il faut toujours passer par ses sections
        - (Section)-[:REFERS_TO]->(Entity) pour certaines sections spécifiques
        - (Ticket)-[:PARENT_TICKET]->(Ticket) pour les relations parent-enfant
        - (Ticket)-[:SIMILAR_TO]-(Ticket) pour les tickets similaires

        Exemples de requêtes valides:
        1. Pour trouver des tickets par résumé:
           ```
           MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section {{type: 'SUMMARY', ticket_id: t.ticket_id}})
           WHERE s.content CONTAINS $summary
           RETURN t.ticket_id
           ```

        2. Pour trouver des tickets avec une solution:
           ```
           MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section {{type: 'SOLUTION', ticket_id: t.ticket_id}})
           RETURN t.ticket_id, s.content
           ```

        Maintenant, génère une requête Cypher qui:
        1. Si des tickets spécifiques sont fournis, concentre-toi sur ces tickets
        2. Sinon, trouve les tickets basés sur les sections pertinentes (utilise HAS_SECTION pour accéder au contenu)
        3. Extrait les sections correspondant aux intentions (SOLUTION, ROOT_CAUSE, etc.)
        4. Inclus les relations pertinentes entre tickets (PARENT_TICKET, SIMILAR_TO)

        IMPORTANT:
        - Utilise TOUJOURS (Ticket)-[:HAS_SECTION]->(Section) pour accéder au contenu
        - Ajoute la condition ticket_id: t.ticket_id quand tu récupères les sections pour garantir de récupérer les bonnes sections
        - N'essaie JAMAIS d'accéder directement à t.summary, t.solution, etc. car ces propriétés n'existent pas
        - Utilise des paramètres pour les valeurs (ex: $summary, $priority)
        - Ne mets PAS de propriétés dans les relations SIMILAR_TO
        - Évite d'utiliser collect() sur des nœuds entiers, préfère des propriétés spécifiques

        Ne génère qu'une seule requête Cypher, concise et optimisée.
        """
        
        try:
            # Appel au LLM (GPT-4o-mini)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un expert en Neo4j et en Cypher. Tu génères des requêtes optimisées et correctes pour extraire des informations d'un graphe de connaissances de tickets d'incident."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            
            # Extraction de la requête Cypher
            cypher_query = response.choices[0].message.content.strip()
            
            # Validation minimale
            if not cypher_query.upper().startswith("MATCH"):
                logger.warning(f"La requête Cypher générée peut être invalide: {cypher_query}")
                # On continue malgré tout, la requête sera testée lors de l'exécution
                
            return cypher_query
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la requête Cypher: {str(e)}")
            # Fallback: méthode basique
            return self._generate_basic_cypher_query(entities, intents, ticket_ids)

    def _generate_basic_cypher_query(self, entities: Dict[str, str], intents: List[str], ticket_ids: List[str] = None) -> str:
        """
        Génère une requête Cypher basique en cas d'échec du LLM.
        Respecte la structure du graphe avec les nœuds Ticket et Section.
        """
        if ticket_ids and len(ticket_ids) > 0:
            # Si des IDs de tickets sont fournis, créer une requête ciblée
            ticket_list = ", ".join([f"'{tid}'" for tid in ticket_ids])
            query = f"""
            MATCH (t:Ticket)
            WHERE t.ticket_id IN [{ticket_list}]
            """
            
            # Ajouter les sections correspondant aux intentions
            for intent in intents:
                section_type = intent
                query += f"""
                OPTIONAL MATCH (t)-[:HAS_SECTION]->(s_{intent.lower()}:Section {{type: '{section_type}', ticket_id: t.ticket_id}})
                """
            
            # Ajouter les relations inter-tickets
            query += """
            OPTIONAL MATCH (t)-[:PARENT_TICKET]->(parent:Ticket)
            OPTIONAL MATCH (t)-[:SIMILAR_TO]-(similar:Ticket)
            """
            
            # Clause RETURN
            return_parts = ["t.ticket_id AS ticket_id"]
            for intent in intents:
                return_parts.append(f"s_{intent.lower()}.content AS {intent.lower()}")
            return_parts.extend(["parent.ticket_id AS parent_id", "collect(similar.ticket_id) AS similar_ids"])
            
            query += f"""
            RETURN {', '.join(return_parts)}
            """
            
            return query
        else:
            # Si aucun ID n'est fourni, générer une requête générique
            return """
            MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section)
            WHERE s.type IN ['SUMMARY', 'DESCRIPTION', 'SOLUTION', 'ROOT_CAUSE']
            RETURN t.ticket_id AS ticket_id, s.type AS section_type, s.content AS content
            LIMIT 10
            """

    def _extract_relevant_subgraph(self, session, ticket_ids: List[str], intents: List[str], entities: Dict[str, str] = None, cypher_query: str = None) -> Dict:
        """
        Extrait les sous-graphes pertinents en fonction des tickets identifiés et des intentions.
        Utilise une requête Cypher générée par le LLM si disponible.
        """
        results = {}
        
        try:
            if cypher_query:
                # Utiliser la requête Cypher générée par le LLM
                params = {}
                
                # Ajouter des paramètres pour les entités si nécessaires
                if entities:
                    for entity_type, entity_value in entities.items():
                        params[entity_type.lower()] = entity_value
                
                # Nettoyer la requête Cypher (supprimer les blocs de code markdown)
                clean_query = cypher_query
                
                # Enlever les délimiteurs de code markdown si présents
                if "```" in clean_query:
                    parts = clean_query.split("```")
                    
                    # Gérer différentes structures de délimiteurs
                    if len(parts) >= 3:
                        # Format standard: ["", "cypher\nQUERY", ""]
                        clean_query = parts[1].replace("cypher", "", 1).strip()
                    elif len(parts) == 2:
                        # Format partiel
                        if parts[0].strip() == "":
                            clean_query = parts[1].strip()
                        else:
                            clean_query = parts[0].strip()
                
                # S'assurer que la requête n'est pas vide et qu'elle commence par MATCH
                if clean_query and clean_query.strip().upper().startswith("MATCH"):
                    try:
                        # Exécuter la requête
                        logger.info(f"Exécution de la requête Cypher générée par LLM: {clean_query}")
                        query_result = session.run(clean_query, params)
                        
                        # Traiter les résultats
                        for record in query_result:
                            ticket_data = dict(record)
                            ticket_id = ticket_data.get("ticket_id")
                            
                            if ticket_id:
                                sections = {}
                                
                                # Extraire les sections des résultats
                                for key, value in ticket_data.items():
                                    if key != "ticket_id" and key in [intent.lower() for intent in intents]:
                                        sections[key.upper()] = value
                                
                                # Ajouter les relations avec d'autres tickets
                                related_tickets = []
                                
                                # Parent tickets
                                if "parent_id" in ticket_data and ticket_data["parent_id"]:
                                    related_tickets.append({
                                        "ticket_id": ticket_data["parent_id"],
                                        "relation_type": "PARENT_OF"
                                    })
                                
                                # Similar tickets
                                if "similar_ids" in ticket_data:
                                    similar_ids = ticket_data["similar_ids"]
                                    if isinstance(similar_ids, list):
                                        for similar_id in similar_ids:
                                            if similar_id:
                                                related_tickets.append({
                                                    "ticket_id": similar_id,
                                                    "relation_type": "SIMILAR_TO"
                                                })
                                
                                results[ticket_id] = {
                                    "ticket_id": ticket_id,
                                    "sections": sections,
                                    "related_tickets": related_tickets
                                }
                        
                        if results:
                            return results
                    
                    except Exception as e:
                        logger.error(f"Erreur lors de l'exécution de la requête Cypher générée par LLM: {str(e)}")
                else:
                    logger.warning("La requête Cypher générée est vide ou invalide")
            
            # Si la requête LLM a échoué ou n'a rien retourné, utiliser la méthode standard
            logger.warning("Utilisation de la méthode standard d'extraction de sous-graphes")
                
            # Méthode standard pour extraire les sous-graphes
            if not ticket_ids:
                logger.warning("Aucun ticket pertinent trouvé")
                return {}
                
            for ticket_id in ticket_ids:
                ticket_info = {"ticket_id": ticket_id, "sections": {}}
                
                # Récupérer les informations de base du ticket
                result = session.run("""
                MATCH (t:Ticket {ticket_id: $ticket_id})-[:HAS_SECTION]->(s:Section {ticket_id: $ticket_id})
                RETURN s.type AS type, s.content AS content
                """, {"ticket_id": ticket_id})
                
                for record in result:
                    section_type = record["type"]
                    content = record["content"]
                    ticket_info["sections"][section_type] = content
                
                # Récupérer les connexions spécifiques comme les tickets similaires, parents, etc.
                related_tickets = []
                
                # Tickets similaires
                similar_result = session.run("""
                MATCH (t1:Ticket {ticket_id: $ticket_id})-[r:SIMILAR_TO]->(t2:Ticket)
                RETURN t2.ticket_id AS related_id, r.similarity AS similarity, 'SIMILAR_TO' AS relation_type
                UNION
                MATCH (t1:Ticket)-[r:SIMILAR_TO]->(t2:Ticket {ticket_id: $ticket_id})
                RETURN t1.ticket_id AS related_id, r.similarity AS similarity, 'SIMILAR_TO' AS relation_type
                """, {"ticket_id": ticket_id})
                
                for record in similar_result:
                    related_tickets.append({
                        "ticket_id": record["related_id"],
                        "relation_type": record["relation_type"],
                        "similarity": record["similarity"]
                    })
                
                # Tickets parents/enfants
                parent_result = session.run("""
                MATCH (t1:Ticket {ticket_id: $ticket_id})-[r:PARENT_TICKET]->(t2:Ticket)
                RETURN t2.ticket_id AS related_id, 'PARENT_OF' AS relation_type
                UNION
                MATCH (t1:Ticket)-[r:PARENT_TICKET]->(t2:Ticket {ticket_id: $ticket_id})
                RETURN t1.ticket_id AS related_id, 'CHILD_OF' AS relation_type
                """, {"ticket_id": ticket_id})
                
                for record in parent_result:
                    related_tickets.append({
                        "ticket_id": record["related_id"],
                        "relation_type": record["relation_type"]
                    })
                
                ticket_info["related_tickets"] = related_tickets
                results[ticket_id] = ticket_info
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des sous-graphes: {str(e)}", exc_info=True)
            return {}

    def _generate_default_response(self, query_text: str) -> Dict:
        """
        Génère une réponse par défaut lorsqu'aucun ticket pertinent n'est trouvé.
        
        Args:
            query_text: Texte de la requête utilisateur
            
        Returns:
            Dictionnaire contenant la réponse par défaut
        """
        try:
            # Tenter de générer une réponse plus personnalisée avec le LLM
            prompt = f"""
            Génère une réponse utile pour un utilisateur qui a posé une question sur un problème technique, 
            mais pour lequel nous n'avons pas trouvé d'informations pertinentes dans notre base de connaissances.
            
            Requête de l'utilisateur: "{query_text}"
            
            Ta réponse doit:
            1. Reconnaître que nous n'avons pas d'information spécifique sur ce problème exact
            2. Suggérer des étapes générales de dépannage qui pourraient être utiles
            3. Proposer des alternatives (comme contacter le support technique)
            4. Demander plus de détails pour raffiner la recherche
            
            Reste professionnel et serviable, tout en reconnaissant les limites de la réponse.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant de support technique expérimenté qui aide à résoudre des problèmes informatiques."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            default_answer = response.choices[0].message.content.strip()
            
            return {
                "answer": default_answer,
                "confidence": 0.2,
                "answer_type": "SOLUTION",
                "relevant_tickets": []
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse par défaut: {str(e)}")
            # En cas d'échec, utiliser une réponse statique
            return {
                "answer": "Je n'ai pas trouvé d'informations pertinentes dans notre base de connaissances pour répondre à votre requête. Veuillez fournir plus de détails ou reformuler votre question.",
                "confidence": 0.2,
                "answer_type": "SOLUTION",
                "relevant_tickets": []
            }

    def _llm_generate_answer(self, subgraphs: Dict, query_text: str, intents: List[str]) -> Dict:
        """
        Utilise un LLM pour générer une réponse structurée basée sur les sous-graphes extraits.
        """
        if not subgraphs:
            return {
                "answer": "Je n'ai pas trouvé d'information pertinente pour cette requête.",
                "confidence": 0.0,
                "relevant_tickets": []
            }
        
        # Préparation des données pour le prompt
        subgraphs_str = json.dumps(subgraphs, indent=2)
        intents_str = json.dumps(intents, indent=2)
        
        prompt = f"""
                Génère une réponse à la requête utilisateur en utilisant les informations extraites du graphe de connaissances.
                
                Requête utilisateur: "{query_text}"
                
                Intentions détectées: {intents_str}
                
                Informations du graphe de connaissances:
                {subgraphs_str}
                
                Instructions:
                1. Commence TOUJOURS ta réponse en indiquant clairement que ta solution est basée sur un ticket existant: "D'après le ticket [ID_TICKET], voici une solution qui pourrait résoudre votre problème:"
                2. Analyse les informations extraites des tickets pertinents
                3. Dans la section "Problème identifié", fais référence au ticket ACTUEL de l'utilisateur: "Votre ticket actuel concerne..." puis explique la similarité avec le cas de référence
                4. Dans la section "Solution proposée", explique en quoi consiste la solution qui a été appliquée dans le cas similaire: "Dans le ticket [ID_TICKET], une solution similaire a été mise en œuvre..."
                5. Formule la solution de manière à l'adapter au problème actuel plutôt que de simplement copier celle du ticket de référence
                6. Si plusieurs tickets contiennent des informations pertinentes, priorise celui qui a la meilleure correspondance
                7. IMPORTANT: Si aucune information vraiment pertinente n'est trouvée, ne pas inventer de solution. Indiquer clairement: "Je ne trouve pas d'information directement pertinente pour votre problème spécifique dans notre base de connaissances, mais voici quelques pistes générales qui pourraient vous aider:"
                
                IMPORTANT: Retourne ta réponse en utilisant strictement le format suivant:
                ```json
                {{
                "sections": [
                    {{"type": "CONTEXT", "content": "D'après le ticket [ID_TICKET], voici une solution qui pourrait résoudre votre problème:"}},
                    {{"type": "SUMMARY", "content": "Votre ticket actuel concerne [description du problème actuel]. Ce cas présente des similarités avec un incident précédent où [brève description du cas de référence]."}},
                    {{"type": "SOLUTION", "content": "Dans le ticket de référence [ID_TICKET], la solution suivante a été appliquée avec succès: [description de la solution adaptée au contexte actuel]"}},
                    {{"type": "STEPS", "content": ["Étape 1: faire ceci", "Étape 2: faire cela", "Étape 3: vérifier que..."]}}
                ]
                }}
                ```
                
                Les sections doivent être parmi: CONTEXT, SUMMARY, SOLUTION, STEPS, WARNING, REFERENCE.
                Pour la section STEPS, utilise une liste d'étapes numérotées.
                Retourne uniquement le JSON sans aucun texte d'introduction ou d'explication.
                """
        
        try:
            # Appel au LLM (GPT-4o-mini)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé en résolution d'incidents techniques. Tu génères des réponses précises et structurées basées sur les données extraites d'un graphe de connaissances de tickets d'incident."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Extraction de la réponse JSON
            try:
                structured_content = json.loads(response.choices[0].message.content.strip())
                
                # S'assurer que la structure est correcte
                if "sections" not in structured_content:
                    structured_content = {"sections": []}
                    
                # Formatage de la réponse complète pour compatibilité
                answer_text = self._format_structured_answer(structured_content["sections"])
                
            except json.JSONDecodeError:
                # Fallback en cas d'échec de parsing JSON
                logger.warning("Échec de parsing JSON depuis LLM, utilisation du texte brut")
                answer_text = response.choices[0].message.content.strip()
                structured_content = {
                    "sections": [
                        {"type": "SOLUTION", "content": answer_text}
                    ]
                }
            
            # Préparation des tickets pertinents pour l'interface
            relevant_tickets = []
            for ticket_id, info in subgraphs.items():
                ticket_data = {
                    "ticket_id": ticket_id,
                    "summary": info["sections"].get("SUMMARY", ""),
                    "solution": info["sections"].get("SOLUTION", ""),
                    "priority": info["sections"].get("PRIORITY", ""),
                    "confidence": info.get("similarity", 0.1)
                }
                relevant_tickets.append(ticket_data)
            
            # Tri des tickets par pertinence réelle
            sorted_tickets = sorted(relevant_tickets, key=lambda x: x["confidence"], reverse=True)

            # Utiliser le score de confiance du ticket le plus pertinent pour la réponse globale
            overall_confidence = sorted_tickets[0]["confidence"] if sorted_tickets else 0.5
            
            # Ajout des sections structurées à la réponse
            return {
                "answer": answer_text,
                "structured_content": structured_content,
                "confidence": overall_confidence,
                "answer_type": intents[0] if intents else "SOLUTION",
                "ticket_id": sorted_tickets[0]["ticket_id"] if sorted_tickets else None,
                "relevant_tickets": sorted_tickets
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse par LLM: {str(e)}")
            # Fallback: générer une réponse basique
            return self._generate_basic_answer(subgraphs, query_text, intents)

    def _format_structured_answer(self, sections: List[Dict]) -> str:
        """
        Formate une réponse textuelle à partir des sections structurées.
        Cette méthode est utilisée pour la compatibilité avec l'existant.
        """
        formatted_text = ""
        
        for section in sections:
            section_type = section.get("type", "")
            content = section.get("content", "")
            
            if section_type == "CONTEXT":
                formatted_text += f"{content}\n\n"
            elif section_type == "SUMMARY":
                formatted_text += f"Résumé : {content}\n\n"
            elif section_type == "SOLUTION":
                formatted_text += f"Solution : {content}\n\n"
            elif section_type == "STEPS":
                formatted_text += "Étapes à suivre :\n"
                if isinstance(content, list):
                    for i, step in enumerate(content, 1):
                        formatted_text += f"{i}. {step}\n"
                else:
                    formatted_text += f"{content}\n"
                formatted_text += "\n"
            elif section_type == "WARNING":
                formatted_text += f"Attention : {content}\n\n"
            elif section_type == "REFERENCE":
                formatted_text += f"Référence : {content}\n\n"
                
        return formatted_text.strip()

    def _generate_basic_answer(self, subgraphs: Dict, query_text: str, intents: List[str]) -> Dict:
        """
        Génère une réponse basique en cas d'échec du LLM.
        """
        # Analyse des sous-graphes pour extraire les informations pertinentes
        primary_info = {}
        for intent in intents:
            if intent == "SOLUTION":
                for ticket_id, info in subgraphs.items():
                    if "SOLUTION" in info["sections"]:
                        solution = info["sections"]["SOLUTION"]
                        if solution:
                            primary_info[ticket_id] = {
                                "content": solution,
                                "type": "SOLUTION",
                                "confidence": 0.9
                            }
            
            elif intent == "ROOT_CAUSE":
                for ticket_id, info in subgraphs.items():
                    if "ROOT_CAUSE" in info["sections"]:
                        root_cause = info["sections"]["ROOT_CAUSE"]
                        if root_cause:
                            primary_info[ticket_id] = {
                                "content": root_cause,
                                "type": "ROOT_CAUSE",
                                "confidence": 0.85
                            }
            
            elif intent == "DESCRIPTION":
                for ticket_id, info in subgraphs.items():
                    if "DESCRIPTION" in info["sections"]:
                        description = info["sections"]["DESCRIPTION"]
                        if description:
                            primary_info[ticket_id] = {
                                "content": description,
                                "type": "DESCRIPTION",
                                "confidence": 0.8
                            }
        
        # Si aucune information primaire n'est trouvée, utiliser le résumé ou toute autre section disponible
        if not primary_info:
            for ticket_id, info in subgraphs.items():
                for section_type in ["SUMMARY", "DESCRIPTION", "SOLUTION", "ROOT_CAUSE"]:
                    if section_type in info["sections"] and info["sections"][section_type]:
                        primary_info[ticket_id] = {
                            "content": info["sections"][section_type],
                            "type": section_type,
                            "confidence": 0.7
                        }
                        break
        
        # Préparation de la réponse
        if primary_info:
            # Trier par confiance
            sorted_info = sorted(
                primary_info.items(), 
                key=lambda x: x[1]["confidence"], 
                reverse=True
            )
            
            best_ticket_id, best_info = sorted_info[0]
            
            # Préparer les tickets pertinents pour l'interface
            relevant_tickets = []
            for ticket_id, info in subgraphs.items():
                similarity_score = info.get("similarity", 0.5)
                ticket_data = {
                    "ticket_id": ticket_id,
                    "summary": info["sections"].get("SUMMARY", ""),
                    "solution": info["sections"].get("SOLUTION", ""),
                    "priority": info["sections"].get("PRIORITY", ""),
                    "confidence": similarity_score
                }
                relevant_tickets.append(ticket_data)
            
            # Trier les tickets pertinents par confiance
            relevant_tickets.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Construire une réponse basique
            answer = f"D'après le ticket {best_ticket_id}, "
            
            if best_info["type"] == "SOLUTION":
                answer += f"voici la solution : {best_info['content']}"
            elif best_info["type"] == "ROOT_CAUSE":
                answer += f"la cause principale est : {best_info['content']}"
            elif best_info["type"] == "DESCRIPTION":
                answer += f"voici la description du problème : {best_info['content']}"
            else:
                answer += f"{best_info['content']}"
            
            return {
                "answer": answer,
                "confidence": best_info["confidence"],
                "answer_type": best_info["type"],
                "ticket_id": best_ticket_id,
                "relevant_tickets": relevant_tickets
            }
        
        return {
            "answer": "Je n'ai pas trouvé d'information spécifique correspondant à votre requête.",
            "confidence": 0.0,
            "relevant_tickets": []
        }

    def get_answer(self, query_text: str) -> Dict:
        """
        Point d'entrée principal pour obtenir une réponse à une requête,
        suivant exactement le workflow décrit dans le papier LinkedIn.
        Utilise la recherche par embeddings comme méthode principale.
        """
        with self.driver.session() as session:
            try:
                # Étape 1: Analyse de la requête pour identifier les entités et intentions (avec LLM)
                entities, intents = self._llm_parse_query_entity_intent(query_text)
                logger.info(f"Entités détectées: {entities}")
                logger.info(f"Intentions détectées: {intents}")
                
                # Étape 2: Recherche des tickets les plus pertinents avec recherche par embeddings
                # C'est la méthode principale décrite dans le papier
                relevant_tickets = self._retrieve_relevant_tickets_embedding(session, entities, query_text, limit=5)
                logger.info(f"Tickets pertinents: {relevant_tickets}")
                
                # Si aucun ticket pertinent n'est trouvé, générer une réponse par défaut
                if not relevant_tickets:
                    logger.warning("Aucun ticket pertinent trouvé, génération d'une réponse par défaut")
                    return self._generate_default_response(query_text)
                
                # Sépare les IDs et les scores de similarité
                relevant_ticket_ids = [ticket_id for ticket_id, _ in relevant_tickets]
                similarity_scores = {ticket_id: score for ticket_id, score in relevant_tickets}
                # Étape 2.5: Générer une requête Cypher avec LLM
                cypher_query = self._llm_generate_graph_query(entities, intents, relevant_ticket_ids)
                
                # Étape 3: Extraction des sous-graphes pertinents avec la requête générée
                subgraphs = self._extract_relevant_subgraph(session, relevant_ticket_ids, intents, entities, cypher_query)

                # Ajouter les scores de similarité aux sous-graphes
                for ticket_id in subgraphs:
                    subgraphs[ticket_id]["similarity"] = similarity_scores.get(ticket_id, 0.0)
                
                # Si aucun sous-graphe n'a été trouvé, utiliser une approche de fallback
                if not subgraphs:
                    logger.warning("Aucun sous-graphe trouvé, utilisation d'une réponse par défaut")
                    return self._generate_default_response(query_text)
                
                # Étape 4: Génération de la réponse avec LLM
                answer = self._llm_generate_answer(subgraphs, query_text, intents)
                
                return answer
                
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de réponse: {str(e)}", exc_info=True)
                return {
                    "answer": f"Une erreur est survenue lors de la recherche d'une réponse. Veuillez réessayer ultérieurement ou reformuler votre question.",
                    "confidence": 0.0,
                    "answer_type": "ERROR",
                    "error": str(e),
                    "relevant_tickets": []
                }

    def close(self):
        """Ferme la connexion à la base de données."""
        self.driver.close()

    def get_system_stats(self):
        """Récupère des statistiques sur le graphe de connaissances."""
        with self.driver.session() as session:
            try:
                stats = {}
                
                # Nombre total de tickets
                result = session.run("MATCH (t:Ticket) RETURN COUNT(t) AS count")
                stats["total_tickets"] = result.single()["count"]
                
                # Nombre total de sections
                result = session.run("MATCH (s:Section) RETURN COUNT(s) AS count")
                stats["total_sections"] = result.single()["count"]
                
                # Distribution des types de sections
                result = session.run("""
                MATCH (s:Section)
                RETURN s.type AS type, COUNT(s) AS count
                ORDER BY count DESC
                """)
                stats["section_types"] = {record["type"]: record["count"] for record in result}
                
                # Nombre de relations intra-ticket
                result = session.run("MATCH ()-[r:HAS_SECTION]->() RETURN COUNT(r) AS count")
                stats["intra_ticket_relations"] = result.single()["count"]
                
                # Nombre de relations inter-ticket explicites (PARENT_TICKET)
                result = session.run("MATCH ()-[r:PARENT_TICKET]->() RETURN COUNT(r) AS count")
                stats["explicit_inter_ticket_relations"] = result.single()["count"]
                
                # Nombre de relations inter-ticket implicites (SIMILAR_TO)
                result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN COUNT(r) AS count")
                stats["implicit_inter_ticket_relations"] = result.single()["count"]
                
                # Nombre de sections avec embeddings
                result = session.run("""
                MATCH (s:Section)
                WHERE s.embedding IS NOT NULL
                RETURN COUNT(s) AS count
                """)
                stats["sections_with_embedding"] = result.single()["count"]
                
                return stats
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des statistiques: {str(e)}")
                return {"error": str(e)}