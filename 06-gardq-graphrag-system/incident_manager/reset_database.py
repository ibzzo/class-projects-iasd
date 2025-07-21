# reset_database_improved.py
import logging
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_reset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_reset")

def reset_neo4j_database():
    """
    Script amélioré pour réinitialiser complètement la base Neo4j et supprimer 
    les anciennes contraintes et index avant de ré-importer les données.
    """
    # Charger les variables d'environnement
    load_dotenv()
    
    # Récupérer les paramètres de connexion à Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'Ticket04#')
    
    try:
        # Connexion à Neo4j
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # 1. Supprimer toutes les données
            logger.info("Suppression de toutes les données...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # 2. Utiliser des commandes plus spécifiques pour supprimer les contraintes
            logger.info("Suppression des contraintes et index existants...")
            
            # Obtenir toutes les contraintes avec leur type
            constraints_result = session.run("SHOW CONSTRAINTS")
            
            # Parcourir toutes les contraintes et les supprimer
            for record in constraints_result:
                try:
                    # On récupère le nom et le type de la contrainte
                    constraint_name = record.get("name", "")
                    constraint_type = record.get("type", "")
                    
                    if constraint_name:
                        # Suppression de la contrainte
                        session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                        logger.info(f"Contrainte supprimée: {constraint_name} (type: {constraint_type})")
                except Exception as e:
                    logger.warning(f"Erreur lors de la suppression de la contrainte: {str(e)}")
            
            # 3. Obtenir et supprimer tous les index
            indexes_result = session.run("SHOW INDEXES")
            
            # Parcourir tous les index et les supprimer
            for record in indexes_result:
                try:
                    # On récupère le nom de l'index
                    index_name = record.get("name", "")
                    
                    if index_name:
                        # Suppression de l'index
                        session.run(f"DROP INDEX {index_name} IF EXISTS")
                        logger.info(f"Index supprimé: {index_name}")
                except Exception as e:
                    logger.warning(f"Erreur lors de la suppression de l'index: {str(e)}")
            
            # 4. Vérification finale
            logger.info("Vérification de la réinitialisation...")
            
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            
            # Nouvelles vérifications des contraintes et index
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            constraint_count = len(constraints)
            
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            index_count = len(indexes)
            
            logger.info(f"État final de la base: {node_count} nœuds, {constraint_count} contraintes, {index_count} index")
            
            if constraint_count > 0:
                logger.warning("Contraintes restantes:")
                for c in constraints:
                    logger.warning(f"  - {c.get('name', 'unknown')}: {c.get('description', '')}")
            
            if index_count > 0:
                logger.warning("Index restants:")
                for i in indexes:
                    logger.warning(f"  - {i.get('name', 'unknown')}: {i.get('description', '')}")
            
            if node_count == 0 and constraint_count == 0 and index_count == 0:
                logger.info("✅ Base de données complètement réinitialisée avec succès")
            else:
                logger.warning("⚠️ La base de données n'est pas complètement vide")
                
                # Dans le cas où il reste des contraintes ou des index, essayer la méthode drastique
                if constraint_count > 0 or index_count > 0:
                    logger.info("Tentative de suppression avec DROP CONSTRAINT et DROP INDEX sans nommer les contraintes/index...")
                    try:
                        session.run("CALL apoc.schema.assert({}, {})")
                        logger.info("Réinitialisation des contraintes et index avec APOC effectuée")
                    except Exception as e:
                        logger.warning(f"APOC non disponible ou erreur: {str(e)}")
                        logger.warning("Si vous n'arrivez pas à supprimer les contraintes/index, essayez de redémarrer le serveur Neo4j.")
        
        # Fermeture de la connexion
        driver.close()
        
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation de la base: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger.info("=== DÉBUT DE LA RÉINITIALISATION AMÉLIORÉE DE LA BASE DE DONNÉES ===")
    reset_neo4j_database()
    logger.info("=== FIN DE LA RÉINITIALISATION DE LA BASE DE DONNÉES ===")