#!/usr/bin/env python3
"""
Script final pour corriger complètement le fichier column_description.csv
"""

def final_fix_column_description():
    # Lire le fichier original
    with open('data/column_description.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Corriger les lignes
    fixed_lines = []
    import re
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line:
            # Trouver la première virgule pour séparer le nom de la colonne et la description
            first_comma_pos = line.find(',')
            if first_comma_pos > 0:
                column_name = line[:first_comma_pos].strip()
                description = line[first_comma_pos+1:].strip()
                
                # Nettoyer la description :
                # 1. Supprimer les patterns comme ",37. SP_500_Distance_to_MA_250d"
                description = re.sub(r',\d+\.\s+\w+.*$', '', description)
                
                # 2. Supprimer les chiffres à la fin (comme "days39" devient "days")
                description = re.sub(r'(\w+)(\d+)$', r'\1', description)
                
                # 3. Supprimer juste les chiffres seuls à la fin
                description = re.sub(r'\d+$', '', description).strip()
                
                # 4. Supprimer les points à la fin qui sont suivis de chiffres
                description = re.sub(r'\.\d+$', '.', description)
                
                # Reconstruire la ligne
                fixed_line = f"{column_name}, {description}"
                fixed_lines.append(fixed_line + '\n')
            else:
                # Si pas de virgule, garder la ligne telle quelle
                fixed_lines.append(line + '\n')
    
    # Écrire le fichier corrigé final
    with open('data/column_description_clean.csv', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fichier final créé: data/column_description_clean.csv")
    print(f"   {len(fixed_lines)} lignes traitées")
    
    # Afficher un échantillon des lignes corrigées
    print("\n📋 Échantillon des lignes corrigées:")
    sample_indices = [0, 36, 37, 38, 39, 40, 95, 96, 113, 114, 132]
    for idx in sample_indices:
        if idx < len(fixed_lines):
            print(f"  Ligne {idx+1}: {fixed_lines[idx].strip()}")
    
    # Test final : essayer de parser en CSV
    print("\n🧪 Test de parsing CSV...")
    try:
        import csv
        with open('data/column_description_clean.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        print(f"✅ Parsing réussi! {len(rows)} lignes lues correctement.")
        
        # Afficher quelques exemples
        print("\n📊 Exemples de colonnes:")
        for i in [36, 37, 38, 39]:
            if i < len(rows):
                print(f"  {rows[i][0]} => {rows[i][1]}")
                
    except Exception as e:
        print(f"❌ Erreur de parsing: {e}")

if __name__ == "__main__":
    final_fix_column_description()