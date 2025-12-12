"""
JIRA-001: Data Loader Module
============================
Module responsable du chargement des données depuis Hugging Face.
Dataset: SetFit/ag_news - Articles d'actualité en 4 catégories.
"""

import pandas as pd
from datasets import load_dataset
from typing import Tuple, Dict


# Mapping des labels numériques vers les noms de catégories
LABEL_MAPPING: Dict[int, str] = {
    0: 'World',
    1: 'Sports', 
    2: 'Business',
    3: 'Sci/Tech'
}


def load_ag_news_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le dataset AG News depuis Hugging Face.
    
    Returns:
        Tuple contenant:
        - df_train: DataFrame des données d'entraînement
        - df_test: DataFrame des données de test
    """
    print("⏳ Chargement du dataset SetFit/ag_news depuis Hugging Face...")
    dataset = load_dataset("SetFit/ag_news")
    
    # Conversion en DataFrames pandas
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    
    # Ajout des noms de catégories
    df_train['category'] = df_train['label'].map(LABEL_MAPPING)
    df_test['category'] = df_test['label'].map(LABEL_MAPPING)
    
    # Ajout d'identifiants uniques
    df_train['article_id'] = [f"train_{i}" for i in range(len(df_train))]
    df_test['article_id'] = [f"test_{i}" for i in range(len(df_test))]
    
    print(f"✅ Chargement terminé!")
    print(f"   - Train: {len(df_train)} articles")
    print(f"   - Test: {len(df_test)} articles")
    
    return df_train, df_test


def load_data(dataset_name: str = "SetFit/ag_news") -> pd.DataFrame:
    """
    Fonction legacy pour compatibilité.
    Charge uniquement les données d'entraînement.
    """
    df_train, _ = load_ag_news_dataset()
    return df_train


def get_label_mapping() -> Dict[int, str]:
    """Retourne le mapping des labels."""
    return LABEL_MAPPING


if __name__ == "__main__":
    # Test du module
    df_train, df_test = load_ag_news_dataset()
    print("\n📊 Aperçu des données:")
    print(df_train.head())