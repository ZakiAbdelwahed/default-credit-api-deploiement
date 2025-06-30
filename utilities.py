import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
import shap
import joblib

NEEDED_COLUMNS = {
    'SK_ID_CURR': 'int64', 'NAME_CONTRACT_TYPE': 'object', 'CODE_GENDER': 'object', 'FLAG_OWN_CAR': 'object', 'FLAG_OWN_REALTY': 'object',
    'CNT_CHILDREN': 'int64', 'AMT_INCOME_TOTAL': 'float64', 'AMT_CREDIT': 'float64', 'AMT_ANNUITY': 'float64', 'AMT_GOODS_PRICE': 'float64',
    'NAME_TYPE_SUITE': 'object', 'NAME_INCOME_TYPE': 'object', 'NAME_EDUCATION_TYPE': 'object', 'NAME_FAMILY_STATUS': 'object', 'NAME_HOUSING_TYPE': 'object',
    'REGION_POPULATION_RELATIVE': 'float64', 'DAYS_BIRTH': 'int64', 'DAYS_EMPLOYED': 'int64', 'DAYS_REGISTRATION': 'float64', 'DAYS_ID_PUBLISH': 'int64',
    'OWN_CAR_AGE': 'float64', 'FLAG_MOBIL': 'int64', 'FLAG_EMP_PHONE': 'int64', 'FLAG_WORK_PHONE': 'int64', 'FLAG_CONT_MOBILE': 'int64',
    'FLAG_PHONE': 'int64', 'FLAG_EMAIL': 'int64', 'OCCUPATION_TYPE': 'object', 'CNT_FAM_MEMBERS': 'float64', 'REGION_RATING_CLIENT': 'int64',
    'REGION_RATING_CLIENT_W_CITY': 'int64', 'WEEKDAY_APPR_PROCESS_START': 'object', 'HOUR_APPR_PROCESS_START': 'int64', 'REG_REGION_NOT_LIVE_REGION': 'int64', 'REG_REGION_NOT_WORK_REGION': 'int64',
    'LIVE_REGION_NOT_WORK_REGION': 'int64', 'REG_CITY_NOT_LIVE_CITY': 'int64', 'REG_CITY_NOT_WORK_CITY': 'int64', 'LIVE_CITY_NOT_WORK_CITY': 'int64', 'ORGANIZATION_TYPE': 'object',
    'EXT_SOURCE_1': 'float64', 'EXT_SOURCE_2': 'float64', 'EXT_SOURCE_3': 'float64', 'APARTMENTS_AVG': 'float64', 'BASEMENTAREA_AVG': 'float64',
    'YEARS_BEGINEXPLUATATION_AVG': 'float64', 'YEARS_BUILD_AVG': 'float64', 'COMMONAREA_AVG': 'float64', 'ELEVATORS_AVG': 'float64', 'ENTRANCES_AVG': 'float64',
    'FLOORSMAX_AVG': 'float64', 'FLOORSMIN_AVG': 'float64', 'LANDAREA_AVG': 'float64', 'LIVINGAPARTMENTS_AVG': 'float64', 'LIVINGAREA_AVG': 'float64',
    'NONLIVINGAPARTMENTS_AVG': 'float64', 'NONLIVINGAREA_AVG': 'float64', 'APARTMENTS_MODE': 'float64', 'BASEMENTAREA_MODE': 'float64', 'YEARS_BEGINEXPLUATATION_MODE': 'float64',
    'YEARS_BUILD_MODE': 'float64', 'COMMONAREA_MODE': 'float64', 'ELEVATORS_MODE': 'float64', 'ENTRANCES_MODE': 'float64', 'FLOORSMAX_MODE': 'float64',
    'FLOORSMIN_MODE': 'float64', 'LANDAREA_MODE': 'float64', 'LIVINGAPARTMENTS_MODE': 'float64', 'LIVINGAREA_MODE': 'float64', 'NONLIVINGAPARTMENTS_MODE': 'float64',
    'NONLIVINGAREA_MODE': 'float64', 'APARTMENTS_MEDI': 'float64', 'BASEMENTAREA_MEDI': 'float64', 'YEARS_BEGINEXPLUATATION_MEDI': 'float64', 'YEARS_BUILD_MEDI': 'float64',
    'COMMONAREA_MEDI': 'float64', 'ELEVATORS_MEDI': 'float64', 'ENTRANCES_MEDI': 'float64', 'FLOORSMAX_MEDI': 'float64', 'FLOORSMIN_MEDI': 'float64',
    'LANDAREA_MEDI': 'float64', 'LIVINGAPARTMENTS_MEDI': 'float64', 'LIVINGAREA_MEDI': 'float64', 'NONLIVINGAPARTMENTS_MEDI': 'float64', 'NONLIVINGAREA_MEDI': 'float64',
    'FONDKAPREMONT_MODE': 'float64', 'HOUSETYPE_MODE': 'object', 'TOTALAREA_MODE': 'float64', 'WALLSMATERIAL_MODE': 'object', 'EMERGENCYSTATE_MODE': 'object',
    'OBS_30_CNT_SOCIAL_CIRCLE': 'float64', 'DEF_30_CNT_SOCIAL_CIRCLE': 'float64', 'OBS_60_CNT_SOCIAL_CIRCLE': 'float64', 'DEF_60_CNT_SOCIAL_CIRCLE': 'float64', 'DAYS_LAST_PHONE_CHANGE': 'float64',
    'FLAG_DOCUMENT_2': 'int64', 'FLAG_DOCUMENT_3': 'int64', 'FLAG_DOCUMENT_4': 'int64', 'FLAG_DOCUMENT_5': 'int64', 'FLAG_DOCUMENT_6': 'int64',
    'FLAG_DOCUMENT_7': 'int64', 'FLAG_DOCUMENT_8': 'int64', 'FLAG_DOCUMENT_9': 'int64', 'FLAG_DOCUMENT_10': 'int64', 'FLAG_DOCUMENT_11': 'int64',
    'FLAG_DOCUMENT_12': 'int64', 'FLAG_DOCUMENT_13': 'int64', 'FLAG_DOCUMENT_14': 'int64', 'FLAG_DOCUMENT_15': 'int64', 'FLAG_DOCUMENT_16': 'int64',
    'FLAG_DOCUMENT_17': 'int64', 'FLAG_DOCUMENT_18': 'int64', 'FLAG_DOCUMENT_19': 'int64', 'FLAG_DOCUMENT_20': 'int64', 'FLAG_DOCUMENT_21': 'int64',
    'AMT_REQ_CREDIT_BUREAU_HOUR': 'float64', 'AMT_REQ_CREDIT_BUREAU_DAY': 'float64', 'AMT_REQ_CREDIT_BUREAU_WEEK': 'float64', 'AMT_REQ_CREDIT_BUREAU_MON': 'float64', 'AMT_REQ_CREDIT_BUREAU_QRT': 'float64',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'float64'
}

# Correction des anomalies de jours + conversion en années
def correction_anomalies_dates(df):
    anomalies = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
    df = df.copy()
    for col in anomalies:
        df[col] = abs(df[col]) / 365  # Conversion en années
    df.rename(columns={
        'DAYS_BIRTH': 'YEARS_BIRTH',
        'DAYS_EMPLOYED': 'YEARS_EMPLOYED',
        'DAYS_REGISTRATION': 'YEARS_REGISTRATION', 
        'DAYS_ID_PUBLISH': 'YEARS_ID_PUBLISH',
        'DAYS_LAST_PHONE_CHANGE': 'YEARS_LAST_PHONE_CHANGE'
    }, inplace=True)
    return df

# Correction des valeurs aberrantes et création d'une colonne pour conserver l'information
def correction_anomalie_years_employed(df):
    df = df.copy()
    valeur_max = df['YEARS_EMPLOYED'].max()
    df['fg_anomalie_years_employed'] = df['YEARS_EMPLOYED'] == valeur_max
    df.loc[df['YEARS_EMPLOYED']==valeur_max, 'YEARS_EMPLOYED'] = np.nan
    return df

# Remplacer 'XNA' par np.nan dans CODE_GENDER
def nettoyage_gender(df):
    df = df.copy()
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', np.nan)
    return df

# Nettoyer les colonnes binaires
def map_col_binaires(df):
    df = df.copy()
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
    df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].map({'Yes': 1, 'No': 0})
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].map({'Cash loans': 1, 'Revolving loans': 0})
    df['CODE_GENDER'] = df['CODE_GENDER'].map({'F': 1, 'M': 0})
    return df

def get_feature_names_from_column_transformer(
    ct: ColumnTransformer, original_features: List[str]
) -> List[str]:
    """
    Récupère les noms des variables après transformation par un ColumnTransformer.

    Paramètres:
        ct: ColumnTransformer utilisé pour les transformations
        original_features: Liste des noms des variables originales

    Retourne:
        Liste des noms des variables transformées
    """
    feature_names = []
    for name, transformer, columns in ct.transformers_:
        if name == "remainder":
            # Gérer les colonnes du remainder (passthrough)
            if transformer == "passthrough":
                remainder_features = [
                    original_features[i] if isinstance(i, int) else i for i in columns
                ]
                feature_names.extend(remainder_features)
            continue
        if hasattr(transformer, "get_feature_names_out"):
            input_features = [
                original_features[i] if isinstance(i, int) else i for i in columns
            ]
            transformed_features = transformer.get_feature_names_out(input_features)
            feature_names.extend(transformed_features)
        else:
            # Pour les transformers sans get_feature_names_out
            feature_names.extend(
                [original_features[i] if isinstance(i, int) else i for i in columns]
            )
    return feature_names


def create_feature_mapping(
    original_features: List[str], transformed_features: List[str]
) -> Dict[str, List[int]]:
    """
    Crée un mapping entre variables originales et leurs indices dans les variables transformées.

    Paramètres:
        original_features: Liste des noms des variables originales
        transformed_features: Liste des noms des variables transformées

    Retourne:
        Dictionnaire mappant chaque variable originale à ses indices transformés
    """
    mapping = {}
    for orig_feature in original_features:
        # Trouver toutes les colonnes transformées qui correspondent à cette variable
        matching_cols = []
        for i, trans_feature in enumerate(transformed_features):
            # Convertir trans_feature en string pour éviter l'erreur TypeError
            trans_feature_str = str(trans_feature)
            # Adapter les patterns selon votre encodage
            if orig_feature in trans_feature_str:
                matching_cols.append(i)
        mapping[orig_feature] = matching_cols
    return mapping


def group_shap_by_original_features(
    shap_values: np.ndarray,
    feature_mapping: Dict[str, List[int]],
    original_features: List[str],
) -> pd.DataFrame:
    """
    Groupe les valeurs SHAP par variables originales en sommant les contributions.
    Paramètres:
    shap_values: Array numpy des valeurs SHAP (n_samples, n_features)
    feature_mapping: Mapping entre variables originales et indices transformés
    original_features: Liste des noms des variables originales
    Retourne:
    DataFrame avec les valeurs SHAP groupées par variable originale
    """

    # Calculer les valeurs shap pour chaque variable
    shap_data = {}
    for orig_feature in original_features:
        if orig_feature in feature_mapping and len(feature_mapping[orig_feature]) > 0:
            col_indices = feature_mapping[orig_feature]
            if len(col_indices) == 1:
                shap_data[orig_feature] = shap_values[:, col_indices[0]]
            else:
                # Plusieurs colonnes : somme des valeurs SHAP
                shap_data[orig_feature] = shap_values[:, col_indices].sum(axis=1)

    return pd.DataFrame(shap_data)


def predict(df, seuil):
    """
    Prédit le risque de défaut de crédit et calcule les valeurs SHAP explicatives
    pour chaque variable et chaque prédiction.
    Paramètres:
    df (pd.DataFrame): Données clients
    Retourne:
    pd.DataFrame: Prédictions (probabilité, classe) et valeurs SHAP par variable originale
    """
    # Pre-processing
    df = correction_anomalies_dates(df).copy()
    df = correction_anomalie_years_employed(df).copy()
    df = nettoyage_gender(df).copy()
    df = map_col_binaires(df).copy()

    # Mettre l'ID du demandeur de prêt en index
    df = df.set_index("SK_ID_CURR").copy()

    # Charger le pipeline contenant les étapes de preprocessing et le modèle
    pipe = joblib.load("model.pkl")

    # Effectuer la prédiction en récupérant les probabilités
    y_pred_df = pd.DataFrame(index=df.index)
    y_pred_df["PROBA_DEFAUT"] = pipe.predict_proba(df)[:, 1]
    y_pred_df["PRED_DEFAUT"] = 0
    y_pred_df.loc[y_pred_df["PROBA_DEFAUT"] > seuil, "PRED_DEFAUT"] = 1

    # Récupérer le nom des variables transformées
    ct = pipe["columntransformer"]
    transformed_features = get_feature_names_from_column_transformer(ct, df.columns)

    # Récupérer le mapping entre variables originales et variables transformées
    feature_mapping = create_feature_mapping(df.columns, transformed_features)

    # Récupérer le modèle
    model = pipe.named_steps["lgbmclassifier"]

    # Transformer X
    X_transformed = pipe[:-1].transform(df)

    # Créer l'explainer avec le modèle
    explainer = shap.TreeExplainer(model)

    # Calculer les valeurs SHAP pour chaque prédiction
    shap_values = explainer.shap_values(X_transformed)

    # Récupérer les valeurs SHAP groupées par variables originales
    shap_df = group_shap_by_original_features(
        shap_values, feature_mapping, df.columns
    ).set_index(df.index)

    # Joindre les prédictions aux valeurs shap par variable
    resultat_df = y_pred_df.merge(
        shap_df, left_index=True, right_index=True, how="left"
    )

    return resultat_df