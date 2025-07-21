# incident_manager/data_loader.py
import os
from dotenv import load_dotenv
import pandas as pd
from .kg_knowledge_system import KGIncidentSystem
import logging

# Charger les variables d'environnement
load_dotenv()
logger = logging.getLogger(__name__)

def load_initial_data(file_path):
    """
    Charge les données initiales depuis le CSV de tickets synthétiques
    """
    try:
        # Vérification des variables d'environnement
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise Exception("Configuration Neo4j manquante dans le fichier .env")

        # Lecture du fichier CSV
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées avec succès. {len(df)} lignes trouvées.")

        # Initialisation du système avec les paramètres de connexion
        # Note: KGIncidentSystem nécessite aussi une clé OpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise Exception("Clé API OpenAI manquante dans le fichier .env")
            
        knowledge_system = KGIncidentSystem(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            openai_api_key=openai_api_key
        )
        
        logger.info(f"Colonnes trouvées : {df.columns.tolist()}")
        # TODO: Adapter la méthode pour KGIncidentSystem
        # knowledge_system.load_historical_data(df)
        logger.warning("Méthode load_historical_data à adapter pour KGIncidentSystem")
        
        return {
            'success': True,
            'message': f'{len(df)} incidents chargés avec succès',
            'total_rows': len(df)
        }

    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {str(e)}")
        return {
            'success': False,
            'message': f"Erreur lors du chargement : {str(e)}"
        }