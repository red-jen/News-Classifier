"""
JIRA-002: Text Preprocessor Module
===================================
Minimal preprocessing for Sentence Transformer embeddings.

IMPORTANT: Sentence Transformers work best with NATURAL TEXT!
- Do NOT remove stopwords (they carry semantic meaning)
- Do NOT remove punctuation (the model handles it)
- Do NOT stem/lemmatize (destroys word meaning)

We only do:
1. Remove duplicates (data quality)
2. Clean whitespace (formatting)
3. Handle missing values (data integrity)
"""

import re
import pandas as pd
from typing import Optional


class TextPreprocessor:
    """
    Minimal text preprocessing for Sentence Transformer embeddings.
    
    Why minimal preprocessing?
    --------------------------
    Sentence Transformers (like paraphrase-multilingual-MiniLM-L12-v2):
    - Have their own tokenizer that handles punctuation
    - Were trained on natural text with stopwords
    - Understand context - "not good" ≠ "good"
    - Removing words can HURT embedding quality
    
    What we do:
    1. Clean whitespace (normalize spaces, trim)
    2. Handle empty/null values
    3. Remove exact duplicates
    
    What we DON'T do (and why):
    - Remove stopwords: "This is NOT good" → "good" (meaning reversed!)
    - Remove punctuation: Model handles it internally
    - Lowercase: Model is case-aware when needed
    - Stemming: "running" → "run" loses tense information
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
        
    def clean_text(self, text: str) -> str:
        """
        Minimal cleaning for embedding models.
        
        Only cleans whitespace and handles edge cases.
        Preserves all semantic content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text (preserving meaning)
        """
        # Handle non-string or empty values
        if not isinstance(text, str) or not text:
            return ""
        
        # Only clean excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Main preprocessing function.
        
        For Sentence Transformers, this is intentionally minimal.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text ready for embedding
        """
        return self.clean_text(text)
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                             text_column: str = 'text',
                             output_column: str = 'text_clean') -> pd.DataFrame:
        """
        Preprocess a text column in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_column: Source column name
            output_column: Output column name
            
        Returns:
            DataFrame with cleaned text column
        """
        df = df.copy()
        print(f"⏳ Preprocessing {len(df)} texts (minimal cleaning for embeddings)...")
        
        # Clean text
        df[output_column] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        empty_count = (df[output_column] == "").sum()
        if empty_count > 0:
            df = df[df[output_column] != ""]
            print(f"   Removed {empty_count} empty texts")
        
        print("✅ Preprocessing complete!")
        return df


def remove_duplicates(df: pd.DataFrame, 
                     text_column: str = 'text') -> pd.DataFrame:
    """
    Remove duplicate articles based on text content.
    
    This is the ONLY heavy preprocessing we do - removing exact duplicates
    is purely a data quality measure, not text transformation.
    
    Args:
        df: DataFrame with articles
        text_column: Column to check for duplicates
        
    Returns:
        DataFrame without duplicates
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=[text_column], keep='first')
    removed_count = initial_count - len(df_clean)
    
    print(f"🗑️ {removed_count} duplicates removed ({initial_count} → {len(df_clean)})")
    return df_clean


# ============================================================================
# LEGACY FUNCTIONS (kept for reference but NOT recommended for embeddings)
# ============================================================================

class LegacyPreprocessor:
    """
    Traditional NLP preprocessing (for TF-IDF, Bag of Words, etc.)
    
    ⚠️ WARNING: Do NOT use these for Sentence Transformer embeddings!
    These are only useful for traditional ML approaches like:
    - TF-IDF vectorization
    - Bag of Words
    - Count Vectorizer
    """
    
    def __init__(self, language: str = 'english'):
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words(language))
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation using regex."""
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords."""
        words = text.split()
        return ' '.join([w for w in words if w.lower() not in self.stop_words])
    
    def remove_numbers(self, text: str) -> str:
        """Remove numbers."""
        return re.sub(r'\d+', '', text)
    
    def lowercase(self, text: str) -> str:
        """Convert to lowercase."""
        return text.lower()


if __name__ == "__main__":
    # Test the module
    preprocessor = TextPreprocessor()
    
    # Test cases showing why minimal preprocessing is better
    test_cases = [
        "Breaking News! The stock market rose 15% today.",
        "This movie is NOT good at all!",
        "The   quick   brown   fox   jumps.",  # Extra whitespace
        "",  # Empty string
        None,  # None value
    ]
    
    print("=" * 60)
    print("MINIMAL PREPROCESSING FOR EMBEDDINGS")
    print("=" * 60)
    
    for text in test_cases:
        cleaned = preprocessor.preprocess(text) if text else preprocessor.preprocess("")
        print(f"Original: {repr(text)}")
        print(f"Cleaned:  {repr(cleaned)}")
        print("-" * 40)