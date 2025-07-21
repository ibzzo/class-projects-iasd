## 📋 Description

GARDQ (Graph Augmented Retrieval for Data Quality) est un système intelligent de gestion des incidents IT utilisant une architecture avancée basée sur les graphes de connaissances (Knowledge Graph) et l'intelligence artificielle. Le projet implémente une approche graphRAG sophistiquée pour améliorer la qualité des données et fournir des suggestions de résolution automatisées aux tickets d'incidents.

### 🎯 Objectifs principaux

- **Automatisation** : Réduire le temps de résolution des incidents récurrents
- **Intelligence** : Utiliser l'IA pour comprendre et suggérer des solutions pertinentes
- **Capitalisation** : Transformer l'historique des incidents en base de connaissances exploitable
- **Performance** : Améliorer l'efficacité des équipes de support IT

## 🏗️ Architecture

### Stack Technologique

| Composant | Technologie | Description |
|-----------|------------|-------------|
| Backend | Django 4.x | Framework web Python |
| Base de données graphe | Neo4j | Stockage et requêtes sur graphe de connaissances |
| IA/ML | OpenAI GPT-4o-mini | Analyse et génération de texte |
| Embeddings | Sentence Transformers | Recherche sémantique multilingue |
| Frontend | Django Templates + D3.js | Interface utilisateur et visualisations |
| Base de données | SQLite | Données Django (utilisateurs, sessions) |

### Architecture du Système

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Utilisateur                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Soumission │  │  Dashboard   │  │  Visualisation   │   │
│  │  d'Incident │  │  Métriques   │  │     Graphe       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Django Application                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              KGIncidentSystem                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │ Analyse │  │Recherche│  │Extraction│  │Génération│   │
│  │  │ Requête │→ │ Tickets │→ │  Graphe  │→ │ Réponse │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     Couche de Données                        │
│  ┌──────────────────┐              ┌──────────────────┐    │
│  │      Neo4j       │              │     OpenAI API    │    │
│  │  Knowledge Graph │              │    GPT-4o-mini    │    │
│  └──────────────────┘              └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Structure du Projet

```

└─  graph-rag/
    ├── incident_manager/          # Application principale Django
    │   ├── models.py             # Modèles de données
    │   ├── views.py              # Contrôleurs et logique métier
    │   ├── kg_knowledge_system.py # Système de graphe de connaissances
    │   ├── data_loader.py        # Chargement des données
    │   ├── migrate_to_kg.py      # Migration vers le système KG
    │   └── management/
    │       └── commands/
    │           └── load_incidents.py # Commande d'import
    │
    ├── incident_system/          # Configuration Django
    │   ├── settings.py          # Paramètres du projet
    │   ├── urls.py              # Routes principales
    │   └── wsgi.py              # Point d'entrée WSGI
    │
    ├── templates/               # Templates HTML
    │   ├── base.html           # Template de base
    │   ├── incident_submission.html # Formulaire de soumission
    │   ├── metrics.html        # Dashboard métriques
    │   └── kg_visualization.html # Visualisation du graphe
    │
    ├── generate_ticket.py      # Script de génération de tickets
    └── manage.py              # Utilitaire Django
```

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8+
- Neo4j 4.x ou supérieur
- Clé API OpenAI
- pip et virtualenv

### Installation

1. **Cloner le projet**

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install django neo4j sentence-transformers openai pandas numpy
```

4. **Configurer Neo4j**
```bash
# Installer Neo4j si nécessaire
# Démarrer Neo4j et noter les credentials
```

5. **Configuration des variables d'environnement**
```bash
# Créer un fichier .env
OPENAI_API_KEY=votre_clé_api
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=votre_mot_de_passe
```

6. **Initialiser la base de données**
```bash
python manage.py migrate
python manage.py createsuperuser
```

7. **Charger les données initiales**
```bash
python manage.py load_incidents chemin/vers/fichier.xlsx // les tickets
```

## 💻 Utilisation

### Démarrer le serveur

```bash
python manage.py runserver
```

L'application sera accessible à `http://localhost:8000`

### Principales fonctionnalités

#### 1. **Soumission d'incident** (`/`)
- Formulaire intuitif pour soumettre un nouvel incident
- Analyse automatique et suggestions basées sur l'historique
- Score de confiance pour chaque suggestion

#### 2. **Dashboard Métriques** (`/metrics/`)
- Statistiques sur le graphe de connaissances
- Nombre de tickets, relations, entités
- Performance du système

#### 3. **Visualisation du Graphe** (`/kg-visualization/`)
- Exploration interactive du graphe de connaissances
- Visualisation des relations entre tickets
- Zoom et navigation dans le graphe

### Pipeline de traitement

1. **Soumission** : L'utilisateur décrit son incident
2. **Analyse** : Le système extrait les entités et intentions
3. **Recherche** : Identification des tickets similaires via embeddings
4. **Extraction** : Récupération du sous-graphe pertinent
5. **Génération** : Synthèse d'une solution contextualisée

## 🔧 Configuration Avancée

### Structure du Graphe de Connaissances

#### Nœuds principaux

- **Ticket** : Incident avec ID unique et métadonnées
- **Section** : Parties structurées (SUMMARY, DESCRIPTION, SOLUTION, etc.)
- **Entity** : Entités métier (Priority, Application, Element, Cause)

#### Relations

- `HAS_SECTION` : Lien Ticket → Section
- `REFERS_TO` : Lien Section → Entity
- `PARENT_TICKET` : Hiérarchie entre tickets
- `SIMILAR_TO` : Similarité calculée entre tickets

### Personnalisation

Le système peut être adapté via :

1. **Modèles d'IA** : Modifier le modèle OpenAI dans `kg_knowledge_system.py`
2. **Prompts** : Ajuster les prompts système pour votre domaine
3. **Embeddings** : Changer le modèle de sentence transformers
4. **Seuils** : Ajuster les seuils de similarité

## 📊 Performance et Optimisations

- **Cache** : Les embeddings sont mis en cache pour éviter les recalculs
- **Index Neo4j** : Index sur les propriétés fréquemment recherchées
- **Batch Processing** : Import de données par lots
- **Recherche hybride** : Combine recherche vectorielle et graphe


## 📝 Licence

Ce projet est développé dans le cadre d'un mémoire académique.

## 👥 Auteur

**Ibrahima Diao**
- Projet de mémoire

---

*Ce projet représente une approche innovante de la gestion des incidents IT, combinant les dernières avancées en IA générative et bases de données graphes pour créer un système de support intelligent et évolutif.*