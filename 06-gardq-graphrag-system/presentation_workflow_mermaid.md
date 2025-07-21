# Diagrammes Mermaid pour la Présentation GARDQ

## Workflow Global

```mermaid
graph LR
    A[Ticket d'Incident] --> B[Analyse LLM<br/>GPT-4]
    B --> C[Construction<br/>du Graphe]
    C --> D[(Neo4j<br/>Knowledge Graph)]
    
    E[Nouvelle Requête] --> F[Analyse de<br/>la Requête]
    F --> G[Recherche<br/>Vectorielle]
    G --> D
    D --> H[Extraction<br/>Sous-graphe]
    H --> I[Contexte<br/>Enrichi]
    I --> J[GPT-4<br/>Génération]
    J --> K[Solution<br/>Suggérée]
    
    style A fill:#e1f5fe
    style E fill:#e1f5fe
    style D fill:#4fc3f7
    style J fill:#81c784
    style K fill:#81c784
```

## Construction du Graphe

```mermaid
graph TD
    A[Ticket Brut] --> B{Analyse GPT-4}
    B --> C[Extraction Sections]
    B --> D[Identification Entités]
    B --> E[Catégorisation]
    
    C --> F[SUMMARY]
    C --> G[DESCRIPTION]
    C --> H[SOLUTION]
    C --> I[ROOT_CAUSE]
    
    D --> J[Application]
    D --> K[Priority]
    D --> L[Cause]
    
    F --> M[Embedding<br/>Vectoriel]
    G --> M
    H --> M
    I --> M
    
    M --> N[(Stockage Neo4j)]
    J --> N
    K --> N
    L --> N
    
    style A fill:#fff3e0
    style B fill:#ffe082
    style N fill:#4fc3f7
```

## Processus de Retrieval

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant S as Système GARDQ
    participant L as LLM (GPT-4)
    participant N as Neo4j
    participant E as Embeddings
    
    U->>S: Nouvelle requête incident
    S->>L: Analyse de la requête
    L-->>S: Entités et intentions
    S->>E: Génération embedding requête
    E-->>S: Vecteur requête
    S->>N: Recherche similarité
    N-->>S: Top-K tickets similaires
    S->>L: Génération requête Cypher
    L-->>S: Requête Cypher optimisée
    S->>N: Extraction sous-graphe
    N-->>S: Contexte enrichi
    S->>L: Prompt + Contexte
    L-->>S: Solution générée
    S->>U: Suggestion de résolution
```

## Architecture de Stockage Neo4j

```mermaid
graph TB
    subgraph "Nœuds Principaux"
        T[Ticket<br/>ticket_id]
        S[Section<br/>section_id<br/>type<br/>content<br/>embedding]
    end
    
    subgraph "Nœuds Entités"
        A[Application<br/>name]
        P[Priority<br/>name]
        C[Cause<br/>name]
        E[Element<br/>name]
    end
    
    T -->|HAS_SECTION| S
    S -->|REFERS_TO| A
    S -->|REFERS_TO| P
    S -->|REFERS_TO| C
    S -->|REFERS_TO| E
    T -->|PARENT_TICKET| T
    T -.->|SIMILAR_TO<br/>similarity: 0.85| T
    
    style T fill:#e3f2fd
    style S fill:#e8f5e9
    style A fill:#fff9c4
    style P fill:#ffebee
    style C fill:#f3e5f5
    style E fill:#fce4ec
```

## Pipeline de Communication LLM

```mermaid
flowchart TD
    A[Contexte du Graphe] --> D[Prompt Engineering]
    B[Requête Utilisateur] --> D
    C[Instructions Système] --> D
    
    D --> E[Prompt Complet]
    E --> F[API GPT-4o-mini]
    
    F --> G{Validation<br/>Réponse}
    G -->|OK| H[Parsing JSON]
    G -->|Erreur| I[Retry avec<br/>Clarification]
    
    H --> J[Solution Structurée]
    I --> F
    
    J --> K[Analyse du Problème]
    J --> L[Étapes de Résolution]
    J --> M[Références Tickets]
    J --> N[Score Confiance]
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style F fill:#81c784
    style J fill:#a5d6a7
```

## Métriques de Performance

```mermaid
graph LR
    subgraph "Retrieval Metrics"
        A[MRR: +78.6%]
        B[Recall@5: +50%]
        C[NDCG@10: +40%]
    end
    
    subgraph "Generation Metrics"
        D[BLEU-4: 0.68]
        E[ROUGE-L: 0.74]
        F[BERTScore: 0.82]
    end
    
    subgraph "Business Impact"
        G[Resolution Time: -28.6%]
        H[First Contact: +35%]
        I[User Satisfaction: +42%]
    end
    
    style A fill:#81c784
    style B fill:#81c784
    style C fill:#81c784
    style G fill:#4fc3f7
    style H fill:#4fc3f7
    style I fill:#4fc3f7
```

## Évolution du Système

```mermaid
timeline
    title Roadmap GARDQ
    
    section Phase 1
        Construction Graphe    : Analyse LLM
                             : Structuration données
                             : Import historique
    
    section Phase 2  
        Retrieval Augmenté   : Embeddings vectoriels
                           : Recherche hybride
                           : Extraction contexte
    
    section Phase 3
        Génération IA       : Prompt engineering
                          : Solutions contextualisées
                          : Scoring confiance
    
    section Phase 4
        Optimisations      : Feedback loop
                         : Apprentissage continu
                         : Prédiction proactive
```

Ces diagrammes peuvent être intégrés dans votre présentation pour illustrer visuellement les différents aspects du système GARDQ.