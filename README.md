# Tâche 2 – Prédiction du type de contenu Netflix (Film ou Série TV)

Classification binaire : prédire si un titre est un **Film** (Movie) ou une **Série TV** (TV Show).

## Méthode
1. Sélection des caractéristiques (année de sortie, date d'ajout, classification, pays, réalisateur connu, nombre de genres, genres)
2. Encodage : One-Hot (classification, pays), standardisation (variables numériques), multi-hot (genres), le tout dans un Pipeline scikit-learn
3. Entraînement : Régression logistique, Arbre de décision, Random Forest, Gradient Boosting
4. Évaluation : accuracy, precision, recall, F1, AUC, validation croisée à 5 plis, matrice de confusion
5. Comparaison des modèles

## Point clé : fuite de données (data leakage)
Les colonnes `duration` (« 90 min » / « 2 Seasons ») et `listed_in` (« Dramas » / « TV Dramas ») révèlent directement la réponse. Elles donnent ~100 % sans rien apprendre. Le notebook le démontre, puis retient un jeu de caractéristiques sans la durée et avec uniquement les genres communs aux films et aux séries.

| Jeu de caractéristiques | Accuracy (CV) |
|---|---|
| A. Durée + genres bruts | 1.000 |
| B. Sans durée, genres bruts | 0.9997 |
| C. Genres normalisés | 0.987 |
| **D. Genres communs (retenu)** | **0.952** |
| Baseline (toujours « Movie ») | 0.697 |

## Résultats (jeu D, 20 % de test)
| Modèle | Accuracy | F1 | AUC |
|---|---|---|---|
| Régression logistique | 0.948 | 0.913 | 0.973 |
| Arbre de décision | 0.948 | 0.913 | 0.947 |
| Random Forest | 0.952 | 0.919 | 0.978 |
| Gradient Boosting | 0.955 | 0.925 | 0.980 |

## Lancer le projet
```bash
pip install -r requirements.txt
jupyter notebook task2_content_type_prediction_EN.ipynb
```
