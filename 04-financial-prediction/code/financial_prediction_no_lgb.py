#!/usr/bin/env python3
"""
Version simplifiée sans LightGBM pour éviter les problèmes de compatibilité
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Chargement des données
print("📊 Chargement des données...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['Dates'] = pd.to_datetime(train_df['Dates'])
test_df['Dates'] = pd.to_datetime(test_df['Dates'])

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Feature engineering simple
print("\n🔧 Feature Engineering...")
for df in [train_df, test_df]:
    df['year'] = df['Dates'].dt.year
    df['month'] = df['Dates'].dt.month
    df['quarter'] = df['Dates'].dt.quarter
    df['day_of_year'] = df['Dates'].dt.dayofyear

# Préparation des données
feature_cols = [col for col in train_df.columns if col not in ['Dates', 'ToPredict']]
X_train = train_df[feature_cols]
y_train = train_df['ToPredict']
X_test = test_df[feature_cols]

# Normalisation
print("\n📐 Normalisation des données...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèles
print("\n🏗️ Entraînement des modèles...")

models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    'CatBoost': CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=False
    ),
    'Ridge': Ridge(alpha=1.0, random_state=42)
}

# Entraînement et prédictions
predictions = {}

for name, model in models.items():
    print(f"  - Entraînement {name}...")
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    predictions[name] = pred
    print(f"    ✅ Moyenne: {pred.mean():.4f}, Std: {pred.std():.4f}")

# Ensemble par moyenne pondérée
print("\n🎯 Création de l'ensemble...")
# Poids basés sur la performance habituelle
weights = {'XGBoost': 0.4, 'CatBoost': 0.4, 'Ridge': 0.2}
final_predictions = np.zeros(len(X_test))

for name, weight in weights.items():
    final_predictions += weight * predictions[name]

# Création du fichier de soumission
print("\n📝 Création du fichier de soumission...")
submission = pd.DataFrame({
    'ID': test_df['Dates'].dt.strftime('%Y-%m-%d'),
    'ToPredict': final_predictions
})

submission.to_csv('submission_simple.csv', index=False)
print("✅ Fichier créé: submission_simple.csv")

# Statistiques
print("\n📊 Statistiques des prédictions:")
print(submission['ToPredict'].describe())

# Graphique
plt.figure(figsize=(12, 5))
plt.plot(test_df['Dates'], final_predictions)
plt.title('Prédictions finales')
plt.xlabel('Date')
plt.ylabel('Valeur prédite')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predictions_simple.png')
plt.show()

print("\n✅ Terminé!")