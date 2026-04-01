"""
Module de gestion des données.
- loader: Chargement depuis HuggingFace
- preprocessor: Nettoyage des textes
"""

from .loader import load_ag_news_dataset, get_label_mapping, LABEL_MAPPING
from .preprocessor import TextPreprocessor, preprocess_dataframe

__all__ = [
    'load_ag_news_dataset',
    'get_label_mapping', 
    'LABEL_MAPPING',
    'TextPreprocessor',
    'preprocess_dataframe'
]