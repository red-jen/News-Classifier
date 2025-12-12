"""
JIRA-005: ChromaDB Storage Module
==================================
Stockage des embeddings dans ChromaDB avec métadonnées.
Collections: train_embeddings, test_embeddings
"""

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class ChromaDBStore:
    """
    Gestionnaire de stockage ChromaDB pour les embeddings.
    
    Permet de:
    - Créer des collections pour train/test
    - Stocker les embeddings avec métadonnées
    - Récupérer les vecteurs pour l'entraînement
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialise le store ChromaDB.
        
        Args:
            persist_directory: Chemin de stockage persistant
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialisation du client ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        self.collections: Dict[str, Any] = {}
        print(f"✅ ChromaDB initialisé: {self.persist_directory}")
        
    def create_collection(self, name: str, 
                         overwrite: bool = False) -> Any:
        """
        Crée ou récupère une collection.
        
        Args:
            name: Nom de la collection
            overwrite: Supprimer si existe déjà
            
        Returns:
            Collection ChromaDB
        """
        if overwrite:
            try:
                self.client.delete_collection(name)
                print(f"🗑️ Collection '{name}' supprimée")
            except Exception:
                pass
                
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.collections[name] = collection
        print(f"✅ Collection '{name}' prête ({collection.count()} éléments)")
        return collection
    
    def add_embeddings(self, collection_name: str,
                      embeddings: np.ndarray,
                      ids: List[str],
                      labels: List[int],
                      categories: List[str],
                      texts: Optional[List[str]] = None) -> None:
        """
        Ajoute des embeddings avec leurs métadonnées.
        
        Args:
            collection_name: Nom de la collection
            embeddings: Array des embeddings
            ids: Identifiants uniques
            labels: Labels numériques
            categories: Noms des catégories
            texts: Textes originaux (optionnel)
        """
        if collection_name not in self.collections:
            self.create_collection(collection_name)
            
        collection = self.collections[collection_name]
        
        # Préparation des métadonnées
        metadatas = [
            {"label": int(label), "category": cat}
            for label, cat in zip(labels, categories)
        ]
        
        # Ajout par batches pour éviter les problèmes de mémoire
        batch_size = 5000
        n_batches = (len(ids) + batch_size - 1) // batch_size
        
        print(f"⏳ Ajout de {len(ids)} embeddings en {n_batches} batches...")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ids))
            
            batch_ids = ids[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx].tolist()
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_docs = texts[start_idx:end_idx] if texts else None
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_docs
            )
            
            print(f"   Batch {i+1}/{n_batches} ajouté")
            
        print(f"✅ {len(ids)} embeddings ajoutés à '{collection_name}'")
    
    def get_embeddings(self, collection_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Récupère tous les embeddings d'une collection.
        
        Args:
            collection_name: Nom de la collection
            
        Returns:
            Tuple (embeddings, labels, ids)
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_collection(collection_name)
            
        collection = self.collections[collection_name]
        
        print(f"⏳ Récupération des embeddings de '{collection_name}'...")
        
        # Récupérer tous les éléments
        results = collection.get(
            include=["embeddings", "metadatas"]
        )
        
        embeddings = np.array(results['embeddings'])
        labels = np.array([m['label'] for m in results['metadatas']])
        ids = results['ids']
        
        print(f"✅ {len(ids)} embeddings récupérés")
        return embeddings, labels, ids
    
    def store_train_test_split(self, 
                               train_embeddings: np.ndarray,
                               train_df: pd.DataFrame,
                               test_embeddings: np.ndarray,
                               test_df: pd.DataFrame,
                               overwrite: bool = True) -> None:
        """
        Stocke les embeddings train et test dans des collections séparées.
        
        Args:
            train_embeddings: Embeddings d'entraînement
            train_df: DataFrame d'entraînement
            test_embeddings: Embeddings de test
            test_df: DataFrame de test
            overwrite: Remplacer les collections existantes
        """
        # Collection d'entraînement
        self.create_collection("train_embeddings", overwrite=overwrite)
        self.add_embeddings(
            "train_embeddings",
            train_embeddings,
            train_df['article_id'].tolist(),
            train_df['label'].tolist(),
            train_df['category'].tolist(),
            train_df['text'].tolist()
        )
        
        # Collection de test
        self.create_collection("test_embeddings", overwrite=overwrite)
        self.add_embeddings(
            "test_embeddings",
            test_embeddings,
            test_df['article_id'].tolist(),
            test_df['label'].tolist(),
            test_df['category'].tolist(),
            test_df['text'].tolist()
        )
        
        print("\n✅ Embeddings train/test stockés dans ChromaDB!")
    
    def list_collections(self) -> List[str]:
        """Liste toutes les collections."""
        return [c.name for c in self.client.list_collections()]


if __name__ == "__main__":
    # Test du module
    store = ChromaDBStore("./test_chromadb")
    
    # Création d'une collection test
    store.create_collection("test_collection", overwrite=True)
    
    # Test d'ajout
    test_embeddings = np.random.rand(10, 384).astype(np.float32)
    test_ids = [f"doc_{i}" for i in range(10)]
    test_labels = [i % 4 for i in range(10)]
    test_categories = ["World", "Sports", "Business", "Sci/Tech"] * 3
    test_categories = test_categories[:10]
    
    store.add_embeddings(
        "test_collection",
        test_embeddings,
        test_ids,
        test_labels,
        test_categories
    )
    
    # Test de récupération
    embeddings, labels, ids = store.get_embeddings("test_collection")
    print(f"\nRécupérés: {len(ids)} embeddings de shape {embeddings.shape}")
