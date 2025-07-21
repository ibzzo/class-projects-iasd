# migrate_to_kg.py
import pandas as pd
import argparse
import logging
from tqdm import tqdm
import os
from dotenv import load_dotenv
from kg_knowledge_system import KGIncidentSystem

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kg_migration")

def main():
    # Charger les variables d'environnement
    load_dotenv()
    
    # Récupérer les paramètres de connexion à Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'Ticket04#')
    
    parser = argparse.ArgumentParser(description='Migrer les données vers le nouveau système Knowledge Graph')
    parser.add_argument('--file', type=str, required=True, help='Chemin vers le fichier Excel contenant les données')
    parser.add_argument('--sheet', type=str, default='MyWorkSheet-1', help='Nom de la feuille Excel à utiliser')
    parser.add_argument('--reset', action='store_true', help='Réinitialiser la base de données avant la migration')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Lecture du fichier Excel: {args.file}, feuille: {args.sheet}")
        df = pd.read_excel(args.file, sheet_name=args.sheet)
        logger.info(f"Données chargées avec succès. {len(df)} enregistrements trouvés.")
        
        # Création d'une instance du nouveau système KG
        kg_system = KGIncidentSystem(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        
        # Réinitialisation de la base si demandé
        if args.reset:
            logger.warning("Réinitialisation de la base de données Neo4j demandée")
            with kg_system.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Base de données réinitialisée avec succès")
                
                # Récréer les contraintes et index après réinitialisation
                kg_system.setup_database()
        
        # Calcul de statistiques avant la migration
        pre_stats = kg_system.get_system_stats()
        logger.info(f"Statistiques avant migration: {pre_stats}")
        
        # Migration des données
        logger.info("Début de la migration des données vers le Knowledge Graph")
        kg_system.load_historical_data(df)
        
        # Calcul de statistiques après la migration
        post_stats = kg_system.get_system_stats()
        logger.info(f"Statistiques après migration: {post_stats}")
        
        # Affichage d'un résumé des changements
        logger.info("Résumé de la migration:")
        tickets_added = post_stats.get('total_tickets', 0) - pre_stats.get('total_tickets', 0)
        sections_added = post_stats.get('total_sections', 0) - pre_stats.get('total_sections', 0)
        intra_relations_added = post_stats.get('intra_ticket_relations', 0) - pre_stats.get('intra_ticket_relations', 0)
        explicit_relations_added = post_stats.get('explicit_inter_ticket_relations', 0) - pre_stats.get('explicit_inter_ticket_relations', 0)
        implicit_relations_added = post_stats.get('implicit_inter_ticket_relations', 0) - pre_stats.get('implicit_inter_ticket_relations', 0)
        
        logger.info(f"  - Tickets ajoutés: {tickets_added}")
        logger.info(f"  - Sections ajoutées: {sections_added}")
        logger.info(f"  - Relations intra-ticket ajoutées: {intra_relations_added}")
        logger.info(f"  - Relations inter-ticket explicites ajoutées: {explicit_relations_added}")
        logger.info(f"  - Relations inter-ticket implicites ajoutées: {implicit_relations_added}")
        
        logger.info("Migration terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la migration: {str(e)}", exc_info=True)
    finally:
        if 'kg_system' in locals():
            kg_system.close()
            logger.info("Fermeture de la connexion Neo4j")

if __name__ == "__main__":
    main()