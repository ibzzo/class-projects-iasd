# Structure du Graphe de Connaissances GARDQ

## Vue d'ensemble

Le système GARDQ utilise Neo4j pour stocker et interroger un graphe de connaissances sophistiqué représentant les tickets d'incidents IT et leurs relations. Cette structure permet une recherche sémantique avancée et une extraction contextuelle d'informations.

## Architecture du Graphe

### 1. Types de Nœuds

#### 🎫 **Ticket**
Le nœud principal représentant un incident IT.

**Propriétés:**
- `ticket_id` (String) : Identifiant unique du ticket

**Contraintes:**
- UNIQUE sur `ticket_id`

#### 📄 **Section**
Représente une partie structurée du contenu d'un ticket.

**Propriétés:**
- `section_id` (String) : ID unique de format `{ticket_id}_{type}`
- `type` (String) : Type de section
- `content` (String) : Contenu textuel
- `ticket_id` (String) : Référence au ticket parent
- `embedding` (List[Float]) : Vecteur d'embedding (optionnel)

**Contraintes:**
- UNIQUE sur (`section_id`, `ticket_id`)

**Types de sections:**
| Type | Description | Embedding |
|------|-------------|-----------|
| SUMMARY | Résumé du ticket | ✅ |
| DESCRIPTION | Description détaillée | ✅ |
| SOLUTION | Solution appliquée | ✅ |
| ROOT_CAUSE | Cause racine identifiée | ✅ |
| PRIORITY | Niveau de priorité | ❌ |
| STATUS | État du ticket | ❌ |
| APPLICATION | Application concernée | ❌ |
| ELEMENT | Élément en résolution | ❌ |
| RESOLUTION_METHOD | Méthode de résolution | ❌ |
| CREATION_DATE | Date de création | ❌ |
| CLOSURE_DATE | Date de clôture | ❌ |

#### 🏷️ **Entités Métier**

**Cause**
- `name` (String) : Nom de la cause
- Contrainte : UNIQUE sur `name`

**Priority**
- `name` (String) : Niveau de priorité (P1, P2, P3, P4)
- Contrainte : UNIQUE sur `name`

**Application**
- `name` (String) : Nom de l'application
- Contrainte : UNIQUE sur `name`

**Element**
- `name` (String) : Élément de résolution
- Contrainte : UNIQUE sur `name`

### 2. Relations

#### ➡️ **HAS_SECTION**
- **De:** Ticket
- **Vers:** Section
- **Description:** Relie un ticket à ses sections
- **Propriétés:** Aucune

#### ➡️ **REFERS_TO**
- **De:** Section
- **Vers:** Entité (Cause, Priority, Application, Element)
- **Description:** Lie une section à une entité métier
- **Propriétés:** Aucune

#### ➡️ **PARENT_TICKET**
- **De:** Ticket (enfant)
- **Vers:** Ticket (parent)
- **Description:** Hiérarchie parent-enfant entre tickets
- **Propriétés:** Aucune

#### ➡️ **SIMILAR_TO**
- **De:** Ticket
- **Vers:** Ticket
- **Description:** Similarité calculée entre tickets
- **Propriétés:** 
  - `similarity` (Float) : Score de similarité cosinus (0-1)

## Schéma Visuel

```
                                    ┌─────────────┐
                                    │   Cause     │
                                    │ name: String│
                                    └─────────────┘
                                           ▲
                                           │ REFERS_TO
                                           │
┌─────────────┐     HAS_SECTION    ┌─────────────────────┐
│   Ticket    │◄───────────────────►│      Section        │
│ ticket_id   │                     │ section_id: String  │
└─────────────┘                     │ type: String        │
      │  ▲                          │ content: String     │
      │  │                          │ ticket_id: String   │
      │  │                          │ embedding: Float[]  │
      │  │                          └─────────────────────┘
      │  │                                 │
      │  │ PARENT_TICKET                   │ REFERS_TO
      │  │                                 ▼
      │  │                          ┌─────────────┐
      │  └──────────────────────────┤  Priority   │
      │                             │ name: String│
      │                             └─────────────┘
      │
      │ SIMILAR_TO                  ┌─────────────┐
      └────────────────────────────►│ Application │
         similarity: Float           │ name: String│
                                    └─────────────┘
                                           ▲
                                           │ REFERS_TO
                                           │
                                    ┌─────────────┐
                                    │   Element   │
                                    │ name: String│
                                    └─────────────┘
```

## Requêtes Cypher Essentielles

### 1. Création d'un Ticket Complet

```cypher
// Créer le ticket
MERGE (t:Ticket {ticket_id: $ticket_id})

// Créer une section avec embedding
CREATE (s:Section {
    section_id: $ticket_id + '_SUMMARY',
    type: 'SUMMARY', 
    content: $summary,
    ticket_id: $ticket_id,
    embedding: $embedding
})

// Relier ticket et section
MATCH (t:Ticket {ticket_id: $ticket_id})
MATCH (s:Section {section_id: $section_id, ticket_id: $ticket_id})
MERGE (t)-[:HAS_SECTION]->(s)

// Créer et relier une priorité
MERGE (p:Priority {name: $priority})
WITH p
MATCH (s:Section {section_id: $priority_section_id, ticket_id: $ticket_id})
MERGE (s)-[:REFERS_TO]->(p)
```

### 2. Recherche par Similarité

```cypher
// Recherche par embedding avec score de similarité
MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section)
WHERE s.embedding IS NOT NULL
WITH t, s, gds.similarity.cosine(s.embedding, $query_embedding) AS similarity
WHERE similarity > $threshold
WITH t, MAX(similarity) AS max_similarity
ORDER BY max_similarity DESC
LIMIT $k
RETURN t.ticket_id, max_similarity
```

### 3. Extraction de Sous-graphe

```cypher
// Extraire toutes les informations d'un ticket
MATCH (t:Ticket {ticket_id: $ticket_id})-[:HAS_SECTION]->(s:Section)
OPTIONAL MATCH (s)-[:REFERS_TO]->(e)
OPTIONAL MATCH (t)-[:PARENT_TICKET]->(parent:Ticket)
OPTIONAL MATCH (t)-[:SIMILAR_TO]-(similar:Ticket)
RETURN 
    t.ticket_id AS ticket,
    collect(DISTINCT {
        type: s.type,
        content: s.content,
        entity: e.name,
        entity_type: labels(e)[0]
    }) AS sections,
    parent.ticket_id AS parent_ticket,
    collect(DISTINCT similar.ticket_id) AS similar_tickets
```

### 4. Analyse des Relations

```cypher
// Trouver les patterns communs entre tickets similaires
MATCH (t1:Ticket)-[r:SIMILAR_TO]-(t2:Ticket)
WHERE r.similarity > 0.8
MATCH (t1)-[:HAS_SECTION]->(:Section)-[:REFERS_TO]->(e1)
MATCH (t2)-[:HAS_SECTION]->(:Section)-[:REFERS_TO]->(e2)
WHERE e1 = e2
RETURN 
    labels(e1)[0] AS entity_type,
    e1.name AS shared_entity,
    count(DISTINCT t1) AS ticket_count
ORDER BY ticket_count DESC
```

## Stratégies d'Optimisation

### 1. Index Neo4j

```cypher
// Index sur les propriétés fréquemment recherchées
CREATE INDEX ticket_id_index FOR (t:Ticket) ON (t.ticket_id);
CREATE INDEX section_composite FOR (s:Section) ON (s.section_id, s.ticket_id);
CREATE INDEX section_type FOR (s:Section) ON (s.type);
CREATE INDEX entity_names FOR (n) ON (n.name) WHERE n:Cause OR n:Priority OR n:Application OR n:Element;
```

### 2. Gestion des Embeddings

- **Stockage**: Les embeddings sont stockés comme listes de floats directement dans Neo4j
- **Dimension**: Typiquement 384 dimensions (sentence-transformers/all-MiniLM-L6-v2)
- **Sections concernées**: SUMMARY, DESCRIPTION, SOLUTION, ROOT_CAUSE uniquement
- **Calcul**: Effectué lors de l'import via le modèle de sentence-transformers

### 3. Patterns de Requêtes

**Pattern 1: Recherche Hybride**
```cypher
// Combiner recherche par mots-clés et similarité sémantique
MATCH (t:Ticket)-[:HAS_SECTION]->(s:Section)
WHERE s.content CONTAINS $keyword OR s.embedding IS NOT NULL
WITH t, s, 
     CASE 
       WHEN s.content CONTAINS $keyword THEN 1.0
       ELSE gds.similarity.cosine(s.embedding, $query_embedding)
     END AS score
WHERE score > $threshold
RETURN t.ticket_id, MAX(score) AS final_score
ORDER BY final_score DESC
```

**Pattern 2: Expansion de Contexte**
```cypher
// Récupérer le contexte élargi d'un ticket
MATCH path = (t:Ticket {ticket_id: $ticket_id})-[:HAS_SECTION|:REFERS_TO|:PARENT_TICKET|:SIMILAR_TO*1..3]-(n)
RETURN path
```

## Points Clés pour l'Évaluation

1. **Métriques de Qualité des Données**
   - Complétude : % de sections remplies par ticket
   - Cohérence : Validité des relations et entités
   - Richesse : Nombre moyen de relations par ticket

2. **Performance de Récupération**
   - Temps de requête moyen
   - Précision de la recherche sémantique
   - Pertinence des tickets similaires

3. **Qualité du Graphe**
   - Densité des connexions
   - Distribution des similarités
   - Couverture des entités métier

Cette structure permet au système GARDQ d'offrir une recherche augmentée par graphe efficace tout en maintenant la qualité et la traçabilité des données.