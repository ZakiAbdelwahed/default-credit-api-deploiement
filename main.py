import os
import pandas as pd
from utilities import (
    predict, 
    correction_anomalies_dates,
    correction_anomalie_years_employed, 
    nettoyage_gender,
    map_col_binaires, 
    convert_to_float, 
    convert_to_df,
    NEEDED_COLUMNS
    )
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Créer l'instance FastAPI avec métadonnées pour la documentation automatique
app = FastAPI(
    title="Prédiction Défaut Crédit API",
    description="API permettant de prédire le risque de défaut d'un client avec le poids de chaque variable",
    version="1.0.0",
)

# Définir le endpoint POST sur la route "/predict"
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint de prédiction pour le risque de défaut de crédit
    Paramètre: file: Fichier CSV contenant les données clients
    Retourne: List[Dict]: Résultats avec prédictions et valeurs SHAP par client
    """

    # Vérifier l'extension du fichier
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail=f"Format de fichier non supporté ({file.filename}). Veuillez fournir un fichier CSV."
        )

    # Lire le fichier uploadé
    df = pd.read_csv(file.file)

    # Vérifier que le fichier n'est pas vide
    if df.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail="Le fichier CSV est vide. Veuillez fournir un fichier contenant des données."
        )

    # Vérifier que le fichier n'est pas trop volumineux
    if df.shape[0] > 1000:
        raise HTTPException(
            status_code=413,
            detail=f"Le fichier est trop volumineux ({df.shape[0]} lignes). Maximum autorisé: 1 000 lignes."
        )
    
    # Vérifier que le fichier contient toutes les colonnes nécessaires    
    col_manquantes = [col for col in NEEDED_COLUMNS.keys() if col not in df.columns]
    if len(col_manquantes) > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier ne contient pas toutes les colonnes nécessaires. Colonnes manquantes: {col_manquantes}"
        )
    
    # Vérifier que le fichier contient au moins l'ID
    if df['SK_ID_CURR'].isna().sum() > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier contient {df['SK_ID_CURR'].isna().sum()} ligne(s) avec des valeurs manquantes dans la colonne SK_ID_CURR. Toutes les lignes doivent avoir un identifiant client."
        )
    
    # Vérifier les types des colonnes
    try:
        df = df.astype(NEEDED_COLUMNS)
    except (ValueError, TypeError) as e:
        # Trouver les colonnes pour lesquelles la conversion a échouée
        failed_columns = []
        for col, dtype in NEEDED_COLUMNS.items():
            if col in df.columns:
                try:
                    df[col].astype(dtype)
                except (ValueError, TypeError):
                    failed_columns.append({
                        'column': col,
                        'expected_type': dtype,
                        'current_type': str(df[col].dtype)
                    })
        
        error_msg = "Impossible de convertir les types de données pour les colonnes: "
        error_msg += ", ".join([f"{fc['column']} (attendu: {fc['expected_type']}, trouvé: {fc['current_type']})" 
                                for fc in failed_columns])
        
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    
    # Effectuer la prédiction
    results_df = predict(df, 0.48)

    # Retourner les résultats sous forme de dictionnaire pour
    # conversion automatique par FastAPI en format JSON
    return results_df.reset_index().to_dict(orient="records")

# Lancer l'API
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)