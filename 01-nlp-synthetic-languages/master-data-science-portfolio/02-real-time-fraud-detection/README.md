# Pipeline de Détection de Fraude en Temps Réel

**Étudiant :** Ibrahim Adiao  
**Email :** diallomari531@gmail.com

Ce projet implémente un système de détection de fraude en temps réel utilisant Kafka et Spark Structured Streaming, suivant exactement les spécifications du wiki hands-on-spark-streaming.

## Architecture

- **Producteur Kafka** : Génère 10-100 transactions par seconde
- **Spark Streaming** : Traite les transactions et détecte les modèles de fraude
- **Dashboard** : Visualisation en temps réel avec Streamlit

## Règles de Détection de Fraude

1. **Transactions à montant élevé** : Signale les transactions > 1000$
2. **Transactions rapides** : Détecte >3 transactions du même utilisateur en <1 minute
3. **Locations multiples** : Détecte les transactions depuis >2 locations en 5 minutes

## Configuration avec Docker

### 1. Démarrer tous les services
```bash
docker-compose up -d
```

Les topics Kafka sont créés automatiquement et tous les services démarrent ensemble.

### 2. Vérifier que tout fonctionne
```bash
docker-compose ps
```

### 3. Visualiser les logs
```bash
# Voir les fraudes détectées
docker logs fraud-detector -f

# Voir les transactions générées
docker logs transaction-producer -f
```

### 4. Accéder au dashboard (optionnel)
Ouvrir dans le navigateur : http://localhost:8501

## Sorties

Le système envoie les alertes de fraude vers :
1. **Console** : Logs en temps réel dans le terminal Spark
2. **Fichiers Parquet** : Répertoire `fraud_alerts_parquet/`
3. **Topic Kafka** : `fraud-alerts` pour traitement ultérieur

## Exemple de Sortie

```
+-------+-------------+-------+--------+-------------------+--------+-----------+----------------------+
|user_id|transaction_id|amount|currency|timestamp         |location|method     |fraud_type            |
+-------+-------------+-------+--------+-------------------+--------+-----------+----------------------+
|u3421  |t-0001234    |2500.0 |EUR     |2025-06-04 10:12:33|Paris   |credit_card|high_value_transaction|
|u5678  |t-0002345    |4500.0 |USD     |2025-06-04 10:13:45|London  |crypto     |high_value_transaction|
|u1234  |t-0003456    |150.0  |GBP     |2025-06-04 10:14:20|Berlin  |paypal     |rapid_transactions    |
+-------+-------------+-------+--------+-------------------+--------+-----------+----------------------+
```

## Test et Vérification

Pour vérifier que le système fonctionne :

1. Consulter les messages Kafka :
```bash
docker exec kafka kafka-console-consumer --topic fraud-alerts --bootstrap-server kafka:29092 --from-beginning
```

2. Vérifier les fichiers Parquet :
```bash
ls -la fraud_alerts_parquet/
```

3. Interface Spark : http://localhost:8081

## Arrêter les Services

```bash
docker-compose down
```

## Technologies Utilisées

- **Apache Kafka** : Streaming de messages
- **Apache Spark** : Traitement en temps réel
- **Docker & Docker Compose** : Conteneurisation et orchestration
- **Python** : Langage de développement
- **Streamlit** : Dashboard web (bonus)

## Structure du Projet

```
fraud-detection-pipeline/
├── README.md               # Ce fichier
├── docker-compose.yml      # Configuration des services
├── producer/              # Générateur de transactions
├── spark/                 # Détecteur de fraude
└── dashboard/             # Interface web
```