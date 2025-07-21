# incident_manager/knowledge_system.py
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class IncidentKnowledgeSystem:
    def __init__(self, uri: str = "neo4j://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "Ticket04#"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.logger = logging.getLogger(__name__)
        self.setup_database()

    def setup_database(self):
        """Configure les contraintes et index dans Neo4j."""
        with self.driver.session() as session:
            try:
                constraints = [
                    "CREATE CONSTRAINT incident_id IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE",
                    "CREATE CONSTRAINT tier2_name IF NOT EXISTS FOR (t:Tier2) REQUIRE t.name IS UNIQUE",
                    "CREATE CONSTRAINT tier3_name IF NOT EXISTS FOR (t:Tier3) REQUIRE t.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Contrainte déjà existante ou erreur: {str(e)}")
                
                logger.info("Base de données configurée avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors de la configuration de la base : {str(e)}")
                raise

    def load_historical_data(self, df: pd.DataFrame):
        """Charge les données des tickets dans le graphe."""
        # Préparation des descriptions pour les embeddings
        descriptions = df.apply(
            lambda x: f"{x['INC Summary']} {x['INC RES Resolution']}", 
            axis=1
        ).fillna('').tolist()
        
        embeddings = self.encoder.encode(descriptions)
        
        with self.driver.session() as session:
            for idx, row in df.iterrows():
                try:
                    # Création du nœud incident
                    session.run("""
                    CREATE (i:Incident {
                        incident_id: $incident_id,
                        summary: $summary,
                        resolution: $resolution,
                        status: $status,
                        submitter: $submitter,
                        assignee: $assignee,
                        duration: $duration,
                        submit_date: $submit_date,
                        closed_date: $closed_date,
                        embedding: $embedding
                    })
                    """, {
                        'incident_id': row['INC ID'],
                        'summary': row['INC Summary'],
                        'resolution': row['INC RES Resolution'],
                        'status': row['INC Status'],
                        'submitter': row['INC DS Submitter'],
                        'assignee': row['AG Assignee'],
                        'duration': float(row['AET Actual Duration (in hours)']),
                        'submit_date': row['INC Submit Date'],
                        'closed_date': row['INC DS Closed Date'],
                        'embedding': embeddings[idx].tolist()
                    })
                    
                    # Création des relations avec les tiers
                    if pd.notna(row['INC Tier 2']):
                        session.run("""
                        MERGE (t:Tier2 {name: $tier2})
                        WITH t
                        MATCH (i:Incident {incident_id: $incident_id})
                        MERGE (i)-[:HANDLED_BY_TIER2]->(t)
                        """, {
                            'tier2': row['INC Tier 2'],
                            'incident_id': row['INC ID']
                        })

                    if pd.notna(row['INC Tier 3']):
                        session.run("""
                        MERGE (t:Tier3 {name: $tier3})
                        WITH t
                        MATCH (i:Incident {incident_id: $incident_id})
                        MERGE (i)-[:HANDLED_BY_TIER3]->(t)
                        """, {
                            'tier3': row['INC Tier 3'],
                            'incident_id': row['INC ID']
                        })

                except Exception as e:
                    logger.error(f"Erreur lors du traitement de l'incident {row['INC ID']}: {str(e)}")
                    continue

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} incidents")

    def find_similar_incidents(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Recherche des incidents similaires."""
        query_embedding = self.encoder.encode(query_text).tolist()
        
        with self.driver.session() as session:
            result = session.run("""
            MATCH (i:Incident)
            WITH i, gds.similarity.cosine(i.embedding, $query_embedding) AS similarity
            WHERE similarity > 0.5
            OPTIONAL MATCH (i)-[:HANDLED_BY_TIER2]->(t2)
            OPTIONAL MATCH (i)-[:HANDLED_BY_TIER3]->(t3)
            RETURN 
                i.incident_id AS incident_id,
                i.summary AS summary,
                i.resolution AS resolution,
                i.status AS status,
                i.duration AS duration,
                t2.name AS tier2,
                t3.name AS tier3,
                similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """, {
                'query_embedding': query_embedding,
                'limit': limit
            })
            
            return [dict(record) for record in result]

    def get_suggested_solution(self, text: str) -> dict:
        # Utiliser la méthode find_similar_incidents pour récupérer les cas similaires
        similar_cases = self.find_similar_incidents(text)
        
        # Pour cet exemple, on choisit la première solution trouvée comme solution recommandée
        if similar_cases:
            recommended_solution = similar_cases[0]['resolution']
            confidence = similar_cases[0]['similarity']
            estimated_time = 1  # Vous pouvez ajuster cette valeur en fonction de votre logique
        else:
            recommended_solution = "Aucune solution trouvée"
            confidence = 0
            estimated_time = 0
        
        return {
            'suggested_solution': recommended_solution,
            'confidence': confidence,
            'estimated_time': estimated_time,
            'similar_cases': similar_cases
        }
    

    def close(self):
        """Ferme la connexion à la base de données."""
        self.driver.close()