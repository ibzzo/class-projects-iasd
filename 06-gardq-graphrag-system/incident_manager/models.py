from django.db import models

# Create your models here.
# incident_manager/models.py
from django.db import models
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
from django.conf import settings

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

class IncidentManager:
    def __init__(self):
        self.db = Neo4jConnection()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        with self.db.driver.session() as session:
            # Création des contraintes et index
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (i:Incident)
                ON (i.incident_id) IS UNIQUE
            """)
            session.run("""
                CALL db.index.fulltext.createNodeIndex(
                    'incidentContent',
                    ['Incident'],
                    ['summary', 'resolution']
                )
            """)

    def import_incidents(self, csv_file):
        df = pd.read_csv(csv_file)
        batch_size = 1000

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            self._process_batch(batch)

    def _process_batch(self, batch):
        with self.db.driver.session() as session:
            # Préparation des embeddings
            summaries = batch['INC Summary'].fillna('').tolist()
            embeddings = self.encoder.encode(summaries)

            for idx, row in batch.iterrows():
                self._create_incident(session, row, embeddings[idx])

    def _create_incident(self, session, row, embedding):
        cypher = """
        MERGE (i:Incident {incident_id: $incident_id})
        SET 
            i.summary = $summary,
            i.resolution = $resolution,
            i.impact = $impact,
            i.status = $status,
            i.submit_date = $submit_date,
            i.embedding = $embedding
        WITH i
        MERGE (g:Group {name: $assigned_group})
        MERGE (i)-[:ASSIGNED_TO]->(g)
        """
        
        session.run(cypher, {
            'incident_id': row['INC Incident ID'],
            'summary': row['INC Summary'],
            'resolution': row['INC RES Resolution'],
            'impact': row['INC Impact'],
            'status': row['INC Status'],
            'submit_date': row['INC Submit Date'],
            'assigned_group': row['AET Assigned Group'],
            'embedding': embedding.tolist()
        })

    def find_similar_incidents(self, query_text, limit=5):
        embedding = self.encoder.encode(query_text).tolist()
        
        with self.db.driver.session() as session:
            result = session.run("""
                MATCH (i:Incident)
                WITH i, gds.similarity.cosine(i.embedding, $embedding) AS similarity
                WHERE similarity > 0.5
                RETURN 
                    i.incident_id AS incident_id,
                    i.summary AS summary,
                    i.resolution AS resolution,
                    i.impact AS impact,
                    i.status AS status,
                    similarity
                ORDER BY similarity DESC
                LIMIT $limit
            """, embedding=embedding, limit=limit)
            
            return [dict(record) for record in result]

    def get_incident_details(self, incident_id):
        with self.db.driver.session() as session:
            result = session.run("""
                MATCH (i:Incident {incident_id: $incident_id})
                OPTIONAL MATCH (i)-[:ASSIGNED_TO]->(g:Group)
                RETURN 
                    i.incident_id AS incident_id,
                    i.summary AS summary,
                    i.resolution AS resolution,
                    i.impact AS impact,
                    i.status AS status,
                    i.submit_date AS submit_date,
                    g.name AS assigned_group
            """, incident_id=incident_id)
            
            return dict(result.single())

    def get_analytics(self):
        with self.db.driver.session() as session:
            result = session.run("""
                MATCH (i:Incident)
                WITH 
                    i.impact AS impact,
                    i.status AS status,
                    COUNT(*) AS count
                RETURN impact, status, count
                ORDER BY count DESC
            """)
            
            return [dict(record) for record in result]