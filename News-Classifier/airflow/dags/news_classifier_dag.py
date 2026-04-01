"""
Pipeline Airflow - News Classifier
Automatise: Chargement → Prétraitement → Embeddings → ChromaDB → Training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os
from pathlib import Path

# Chemin pour Docker (/app) ou local
if os.path.exists("/app/src"):
    PROJECT_PATH = Path("/app")  # Docker
else:
    PROJECT_PATH = Path(__file__).parent.parent.parent  # Local

sys.path.insert(0, str(PROJECT_PATH / "src"))
sys.path.insert(0, str(PROJECT_PATH))

default_args = {
    'owner': 'data_team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ==================== TÂCHES ====================

def task_load_data(**context):
    """Télécharge AG News depuis HuggingFace"""
    from src.data.loader import load_ag_news_dataset
    
    df_train, df_test = load_ag_news_dataset()
    df_train.to_csv(str(PROJECT_PATH / "data/raw/train.csv"), index=False)
    df_test.to_csv(str(PROJECT_PATH / "data/raw/test.csv"), index=False)
    print(f"✅ train={len(df_train)}, test={len(df_test)}")


def task_preprocess(**context):
    """Nettoie les textes"""
    import pandas as pd
    from src.data.preprocessor import TextPreprocessor
    
    df_train = pd.read_csv(str(PROJECT_PATH / "data/raw/train.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/raw/test.csv"))
    
    preprocessor = TextPreprocessor()
    df_train = preprocessor.preprocess_dataframe(df_train)
    df_test = preprocessor.preprocess_dataframe(df_test)
    
    df_train.to_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"), index=False)
    df_test.to_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"), index=False)
    print(f"✅ Nettoyé: train={len(df_train)}, test={len(df_test)}")


def task_embeddings(**context):
    """Génère les embeddings 384D"""
    import pandas as pd
    import numpy as np
    from src.embeddings.generator import EmbeddingGenerator
    
    df_train = pd.read_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"))
    
    generator = EmbeddingGenerator()
    train_emb = generator.generate_embeddings_df(df_train)
    test_emb = generator.generate_embeddings_df(df_test)
    
    np.save(str(PROJECT_PATH / "data/processed/train_embeddings.npy"), train_emb)
    np.save(str(PROJECT_PATH / "data/processed/test_embeddings.npy"), test_emb)
    print(f"✅ Embeddings: train={train_emb.shape}, test={test_emb.shape}")


def task_chromadb(**context):
    """Stocke dans ChromaDB"""
    import pandas as pd
    import numpy as np
    from src.storage.chroma_store import ChromaDBStore
    
    df = pd.read_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"))
    embeddings = np.load(str(PROJECT_PATH / "data/processed/train_embeddings.npy"))
    
    store = ChromaDBStore(str(PROJECT_PATH / "data/chromadb"))
    store.store_embeddings(embeddings, df)
    print("✅ Stocké dans ChromaDB")


def task_train(**context):
    """Entraîne Logistic Regression"""
    import numpy as np
    import pandas as pd
    from src.models.classifier import NewsClassifier
    
    train_emb = np.load(str(PROJECT_PATH / "data/processed/train_embeddings.npy"))
    test_emb = np.load(str(PROJECT_PATH / "data/processed/test_embeddings.npy"))
    df_train = pd.read_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"))
    
    classifier = NewsClassifier('logistic_regression')
    classifier.train(train_emb, df_train['label'].values)
    results = classifier.evaluate(test_emb, df_test['label'].values)
    classifier.save_model(str(PROJECT_PATH / "models/best_classifier.joblib"))
    print(f"✅ Accuracy: {results['accuracy']:.2%}")


# ==================== DAG ====================

with DAG(
    dag_id='news_classifier_pipeline',
    default_args=default_args,
    description='Pipeline NLP: HuggingFace → Embeddings → ML',
    schedule_interval='@weekly',
    catchup=False,
    tags=['nlp', 'ml']
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    load = PythonOperator(task_id='load_data', python_callable=task_load_data)
    preprocess = PythonOperator(task_id='preprocess', python_callable=task_preprocess)
    embeddings = PythonOperator(task_id='embeddings', python_callable=task_embeddings)
    chromadb = PythonOperator(task_id='chromadb', python_callable=task_chromadb)
    train = PythonOperator(task_id='train', python_callable=task_train)
    
    # Flux: start → load → preprocess → embeddings → chromadb → train → end
    start >> load >> preprocess >> embeddings >> chromadb >> train >> end
