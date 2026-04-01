

import numpy as np
import pandas as pd
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm




def loaddata(str):

   








class EmbeddingGenerator:
   
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialise le générateur d'embeddings.
        
        Args:
            model_name: Nom du modèle Sentence Transformers à utiliser
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: Optional[int] = None
        
    def load_model(self) -> None:
        """Charge le modèle Sentence Transformers."""
        print(f"⏳ Chargement du modèle {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modèle chargé! Dimension des embeddings: {self.embedding_dim}")
        
    def generate_embeddings(self, texts: List[str], 
                           batch_size: int = 32,
                           show_progress: bool = True) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes.
        
        Args:
            texts: Liste des textes à encoder
            batch_size: Taille des batches pour le traitement
            show_progress: Afficher la barre de progression
            
        Returns:
            Array numpy de shape (n_texts, embedding_dim)
        """
        if self.model is None:
            self.load_model()
            
        print(f"⏳ Génération des embeddings pour {len(texts)} textes...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"✅ Embeddings générés! Shape: {embeddings.shape}")
        return embeddings
    
    def generate_embeddings_df(self, df: pd.DataFrame,
                               text_column: str = 'text_clean',
                               batch_size: int = 32) -> np.ndarray:
        """
        Génère les embeddings à partir d'un DataFrame.
        
        Args:
            df: DataFrame contenant les textes
            text_column: Nom de la colonne de texte
            batch_size: Taille des batches
            
        Returns:
            Array numpy des embeddings
        """
        texts = df[text_column].tolist()
        return self.generate_embeddings(texts, batch_size)
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim


if __name__ == "__main__":
    # Test du module
    generator = EmbeddingGenerator()
    
    sample_texts = [
        "The stock market reached new highs today",
        "The football team won the championship",
        "New AI technology revolutionizes healthcare"
    ]
    
    embeddings = generator.generate_embeddings(sample_texts)
    print(f"\nExemple d'embedding (5 premières valeurs): {embeddings[0][:5]}")
