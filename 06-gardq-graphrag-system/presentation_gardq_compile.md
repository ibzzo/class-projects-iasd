# Instructions pour Compiler la Présentation LaTeX sur Overleaf

## 1. Créer un Nouveau Projet sur Overleaf

1. Allez sur [Overleaf.com](https://www.overleaf.com)
2. Cliquez sur "New Project" → "Blank Project"
3. Nommez le projet "GARDQ Presentation"

## 2. Uploader le Fichier

1. Supprimez le fichier `main.tex` par défaut
2. Uploadez `presentation_gardq.tex`
3. Renommez-le en `main.tex` (ou changez le fichier principal dans les paramètres)

## 3. Configuration du Compilateur

Dans les paramètres du projet (icône engrenage) :
- **Compiler**: XeLaTeX ou LuaLaTeX (pour un meilleur support UTF-8)
- **Main document**: main.tex

## 4. Packages Nécessaires

La présentation utilise les packages suivants (déjà inclus dans le code) :
- `beamer` avec le thème `metropolis`
- `tikz` pour les schémas
- `pgfplots` pour les graphiques
- `listings` pour le code
- `fontawesome` pour les icônes

## 5. Personnalisations Possibles

### Couleurs
Les couleurs GARDQ sont définies au début :
```latex
\definecolor{gardqblue}{RGB}{79, 195, 247}
\definecolor{gardqgreen}{RGB}{129, 199, 132}
\definecolor{gardqorange}{RGB}{255, 183, 77}
```

### Logo Neo4j
Pour le slide sur Neo4j, vous pouvez :
1. Uploader une image `neo4j-icon.png`
2. Ou remplacer `\includegraphics[width=0.5cm]{neo4j-icon.png}` par du texte

### Ajuster la Taille
Si les schémas sont trop grands/petits, modifiez le paramètre `scale` :
```latex
\begin{tikzpicture}[scale=0.8, transform shape]
```

## 6. Résolution de Problèmes

### Si le thème Metropolis n'est pas disponible
Remplacez :
```latex
\usetheme{metropolis}
```
Par :
```latex
\usetheme{Madrid}  % ou un autre thème standard
```

### Si les icônes FontAwesome ne s'affichent pas
Commentez la ligne :
```latex
% \usepackage{fontawesome}
```
Et remplacez les icônes par du texte :
- `\faCheckCircle{}` → `✓`
- `\faRobot{}` → `[AI]`
- etc.

## 7. Export

Une fois compilé avec succès :
1. Cliquez sur "Download PDF"
2. Ou partagez directement le lien Overleaf

## Notes

- La présentation contient 19 slides au total
- Les schémas TikZ sont optimisés pour un format 16:9
- Le code est commenté pour faciliter les modifications
- Les graphiques utilisent des données réelles du papier LinkedIn