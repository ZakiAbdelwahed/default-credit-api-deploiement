# Defaut Credit API

Une API FastAPI pour la prédiction du risque de défaut de crédit utilisant LightGBM et SHAP pour l'explicabilité.

## 📋 Description

Cette API permet de prédire le risque de défaut de crédit à partir de fichiers CSV contenant les données clients. Elle utilise un modèle LightGBM pré-entraîné et fournit des explications SHAP pour chaque prédiction, permettant de comprendre l'importance de chaque variable dans la décision.

## 🚀 Fonctionnalités

- **Prédiction de défaut de crédit** : Probabilité et classification binaire (défaut/non-défaut)
- **Explicabilité IA** : Valeurs SHAP pour chaque variable et chaque prédiction
- **Validation robuste** : Vérification des types, colonnes manquantes et format des données
- **Traitement par lots** : Jusqu'à 1000 clients par fichier CSV

## 🛠️ Technologies utilisées

- **Framework** : FastAPI 0.115.12
- **Machine Learning** : LightGBM 4.6.0
- **Explicabilité** : SHAP 0.46.0
- **Preprocessing** : Scikit-learn 1.6.1, Category-encoders 2.8.1
- **Data manipulation** : Pandas 2.2.3, NumPy 1.24.4
- **Serveur** : Uvicorn 0.34.2

## 📊 Format des données requises

Le fichier CSV doit contenir **121 colonnes** avec les types définies dans `NEEDED_COLUMNS`

## 📚 Utilisation de l'API

### Documentation interactive

- **Local** : `http://localhost:8000/docs`
- **En ligne** : https://defaut-credit-api.onrender.com/docs

### Endpoint de prédiction

**Exemple avec Python :**

```python
import pandas as pd
import requests

# Définir l'URL de l'API et le fichier à analyser
file_path = "fichiers tests/application_test_2_clients.csv"
url = "https://defaut-credit-api.onrender.com/predict"

# Ouvrir le fichier et l'envoyer à l'API
with open(file_path, 'rb') as f:
    files = {'file': (file_path, f, 'text/csv')}
    response = requests.post(url, files=files)

# Récupérer et afficher les résultats
results = response.json()
results_df = pd.DataFrame(results)
print(results_df)
```

**Explication des résultats :**
- `PROBA_DEFAUT` : Probabilité de défaut (0 à 1)
- `PRED_DEFAUT` : Prédiction binaire (0=pas de défaut, 1=défaut)
- Autres colonnes : Valeurs SHAP expliquant l'importance de chaque variable

## ⚙️ Configuration

### Seuil de classification

Le seuil par défaut est fixé à **0.48** dans le code (`predict(df, 0.48)`). Pour le modifier :

```python
# Dans main.py, ligne 69
results_df = predict(df, 0.48)  # Changer la valeur ici
```

## 🤖 Modèle ML

### Preprocessing automatique

1. **Correction des dates** : Conversion des jours en années
2. **Gestion des anomalies** : `YEARS_EMPLOYED` (valeurs aberrantes)
3. **Nettoyage** : Remplacement des valeurs 'XNA' par NaN
4. **Encodage** : Variables binaires et catégorielles

### Explicabilité SHAP

Chaque prédiction inclut les valeurs SHAP pour toutes les variables, permettant de comprendre :
- Quelles variables influencent le plus la décision
- Dans quel sens (positif/négatif) chaque variable contribue
- L'importance relative de chaque facteur

## 📁 Structure du projet

```
default-credit-api-deploiement/
├── main.py              # Point d'entrée FastAPI
├── utilities.py         # Fonctions de preprocessing et prédiction
├── requirements.txt     # Dépendances Python
├── model.pkl           # Modèle LightGBM pré-entraîné (à ajouter)
├── Dockerfile          # Configuration Docker (optionnel)
└── README.md          # Ce fichier
```