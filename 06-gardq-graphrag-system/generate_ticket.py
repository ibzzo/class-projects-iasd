import openai
import pandas as pd
import random
from datetime import datetime, timedelta

# Clé API OpenAI (remplace par ta propre clé)
openai.api_key = "TA_CLE_API"

# Fonction pour générer un ticket
def generate_ticket():
    prompt = """Génère un ticket IT sous la forme JSON :
    {
      "INC Summary": "Brève description du problème",
      "INC RES Resolution": "Solution détaillée",
      "INC Tier 2": "Catégorie principale",
      "INC Tier 3": "Sous-catégorie",
      "INC Status": "Statut (Ouvert, Résolu, En attente)",
      "INC DS Submitter": "Nom fictif",
      "AG Assignee": "Nom de l'agent assigné",
      "AET Actual Duration (in hours)": "Durée en heures",
      "INC Submit Date": "Date de soumission (AAAA-MM-JJ)",
      "INC DS Closed Date": "Date de fermeture (AAAA-MM-JJ)"
    }
    """
   
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Tu es un assistant spécialisé en ITSM chez Orange."},
                  {"role": "user", "content": prompt}]
    )
    
    return eval(response["choices"][0]["message"]["content"])

# Génération de 100 tickets synthétiques
data = [generate_ticket() for _ in range(1000)]

# Convertir en DataFrame
df = pd.DataFrame(data)

# Sauvegarde CSV
df.to_csv("tickets_synthetiques.csv", index=False)
