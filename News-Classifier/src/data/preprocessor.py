"""
JIRA-002: Text Preprocessor Module
===================================
Module de prétraitement des textes pour la classification d'articles.
Inclut: normalisation, suppression doublons, stopwords, ponctuation.
"""

import re
import pandas as pd
from typing import List, Set
import nltk

# Télécharger les ressources NLTK si nécessaire
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords


class TextPreprocessor:
    """
    Classe pour le prétraitement des textes d'articles d'actualité.
    
    Étapes de prétraitement:
    1. Normalisation (minuscules)
    2. Suppression de la ponctuation (regex)
    3. Suppression des chiffres
    4. Suppression des stopwords
    5. Nettoyage des espaces
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialise le préprocesseur.
        
        Args:
            language: Langue pour les stopwords (default: 'english')
        """
        self.language = language
        self.stop_words: Set[str] = set(stopwords.words(language))
        
    def normalize_text(self, text: str) -> str:
        """
        Normalise le texte en minuscules.
        
        Args:
            text: Texte à normaliser
            
        Returns:
            Texte en minuscules
        """
        if not isinstance(text, str):
            return ""
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """
        Supprime la ponctuation avec des expressions régulières.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte sans ponctuation
        """
        # Supprime tous les caractères non-alphanumériques sauf espaces
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def remove_numbers(self, text: str) -> str:
        """
        Supprime les chiffres du texte.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte sans chiffres
        """
        return re.sub(r'\d+', '', text)
    
    def remove_stopwords(self, text: str) -> str:
        """
        Supprime les stopwords du texte.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte sans stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def clean_whitespace(self, text: str) -> str:
        """
        Nettoie les espaces multiples.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte avec espaces normalisés
        """
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess(self, text: str) -> str:
        """
        Applique toutes les étapes de prétraitement.
        
        Args:
            text: Texte brut à prétraiter
            
        Returns:
            Texte prétraité
        """
        text = self.normalize_text(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_stopwords(text)
        text = self.clean_whitespace(text)
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                             text_column: str = 'text',
                             output_column: str = 'text_clean') -> pd.DataFrame:
        """
        Prétraite une colonne de texte dans un DataFrame.
        
        Args:
            df: DataFrame contenant les textes
            text_column: Nom de la colonne source
            output_column: Nom de la colonne de sortie
            
        Returns:
            DataFrame avec la colonne prétraitée
        """
        df = df.copy()
        print(f"⏳ Prétraitement de {len(df)} textes...")
        df[output_column] = df[text_column].apply(self.preprocess)
        print("✅ Prétraitement terminé!")
        return df


def remove_duplicates(df: pd.DataFrame, 
                     text_column: str = 'text') -> pd.DataFrame:
    """
    Supprime les articles en double basé sur le texte.
    
    Args:
        df: DataFrame avec les articles
        text_column: Colonne à utiliser pour détecter les doublons
        
    Returns:
        DataFrame sans doublons
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=[text_column], keep='first')
    removed_count = initial_count - len(df_clean)
    
    print(f"🗑️ {removed_count} doublons supprimés ({initial_count} → {len(df_clean)})")
    return df_clean


if __name__ == "__main__":
    # Test du module
    preprocessor = TextPreprocessor()
    
    sample_text = "Breaking News! The stock market rose 15% today, amazing results!!!"
    print(f"Original: {sample_text}")
    print(f"Prétraité: {preprocessor.preprocess(sample_text)}")