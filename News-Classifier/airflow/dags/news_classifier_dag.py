"""
JIRA-008: Apache Airflow DAG
=============================
Orchestration complète du pipeline NLP pour la classification d'actualités.
Ce DAG automatise l'ensemble du processus de bout en bout.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import sys
from pathlib import Path

# Ajouter le chemin du projet
PROJECT_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_PATH / "src"))


# Configuration par défaut du DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def task_load_data(**context):
    """
    JIRA-001: Chargement des données depuis Hugging Face.
    """
    from src.data.loader import load_ag_news_dataset
    
    print("📥 Chargement des données AG News...")
    df_train, df_test = load_ag_news_dataset()
    
    # Sauvegarder en CSV pour les tâches suivantes
    df_train.to_csv(str(PROJECT_PATH / "data/raw/train.csv"), index=False)
    df_test.to_csv(str(PROJECT_PATH / "data/raw/test.csv"), index=False)
    
    print(f"✅ Données sauvegardées: train={len(df_train)}, test={len(df_test)}")
    
    return {'train_size': len(df_train), 'test_size': len(df_test)}


def task_preprocess_data(**context):
    """
    JIRA-002: Prétraitement des textes.
    """
    import pandas as pd
    from src.data.preprocessor import TextPreprocessor, remove_duplicates
    
    print("🔧 Prétraitement des données...")
    
    # Charger les données
    df_train = pd.read_csv(str(PROJECT_PATH / "data/raw/train.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/raw/test.csv"))
    
    # Prétraitement
    preprocessor = TextPreprocessor()
    
    df_train = remove_duplicates(df_train)
    df_test = remove_duplicates(df_test)
    
    df_train = preprocessor.preprocess_dataframe(df_train)
    df_test = preprocessor.preprocess_dataframe(df_test)
    
    # Sauvegarder
    df_train.to_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"), index=False)
    df_test.to_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"), index=False)
    
    print(f"✅ Prétraitement terminé: train={len(df_train)}, test={len(df_test)}")
    
    return {'train_clean': len(df_train), 'test_clean': len(df_test)}


def task_generate_embeddings(**context):
    """
    JIRA-004: Génération des embeddings avec Sentence Transformers.
    """
    import pandas as pd
    import numpy as np
    from src.embeddings.generator import EmbeddingGenerator
    
    print("🧠 Génération des embeddings...")
    
    # Charger les données prétraitées
    df_train = pd.read_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"))
    
    # Générer les embeddings
    generator = EmbeddingGenerator()
    
    train_embeddings = generator.generate_embeddings_df(df_train)
    test_embeddings = generator.generate_embeddings_df(df_test)
    
    # Sauvegarder les embeddings
    np.save(str(PROJECT_PATH / "data/processed/train_embeddings.npy"), train_embeddings)
    np.save(str(PROJECT_PATH / "data/processed/test_embeddings.npy"), test_embeddings)
    
    print(f"✅ Embeddings générés: train={train_embeddings.shape}, test={test_embeddings.shape}")
    
    return {'train_shape': train_embeddings.shape, 'test_shape': test_embeddings.shape}


def task_store_chromadb(**context):
    """
    JIRA-005: Stockage des embeddings dans ChromaDB.
    """
    import pandas as pd
    import numpy as np
    from src.storage.chroma_store import ChromaDBStore
    
    print("💾 Stockage dans ChromaDB...")
    
    # Charger les données
    df_train = pd.read_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"))
    train_embeddings = np.load(str(PROJECT_PATH / "data/processed/train_embeddings.npy"))
    test_embeddings = np.load(str(PROJECT_PATH / "data/processed/test_embeddings.npy"))
    
    # Stocker dans ChromaDB
    store = ChromaDBStore(str(PROJECT_PATH / "data/chromadb"))
    store.store_train_test_split(train_embeddings, df_train, test_embeddings, df_test)
    
    print("✅ Embeddings stockés dans ChromaDB")
    
    return {'status': 'success'}


def task_train_model(**context):
    """
    JIRA-006: Entraînement et évaluation des modèles ML.
    """
    import numpy as np
    import pandas as pd
    from src.models.classifier import NewsClassifier, compare_models
    
    print("🎯 Entraînement des modèles...")
    
    # Charger les embeddings
    train_embeddings = np.load(str(PROJECT_PATH / "data/processed/train_embeddings.npy"))
    test_embeddings = np.load(str(PROJECT_PATH / "data/processed/test_embeddings.npy"))
    
    df_train = pd.read_csv(str(PROJECT_PATH / "data/processed/train_clean.csv"))
    df_test = pd.read_csv(str(PROJECT_PATH / "data/processed/test_clean.csv"))
    
    y_train = df_train['label'].values
    y_test = df_test['label'].values
    
    # Comparer les modèles
    results = compare_models(train_embeddings, y_train, test_embeddings, y_test)
    
    # Entraîner le meilleur modèle (Logistic Regression généralement)
    best_model = NewsClassifier('logistic_regression')
    best_model.train(train_embeddings, y_train)
    best_model.evaluate(test_embeddings, y_test)
    best_model.check_overfitting(train_embeddings, y_train, test_embeddings, y_test)
    
    # Sauvegarder le modèle
    best_model.save_model(str(PROJECT_PATH / "models/best_model.joblib"))
    
    print("✅ Modèle entraîné et sauvegardé")
    
    return {'best_accuracy': float(results.iloc[0]['test_accuracy'])}


def task_validate_pipeline(**context):
    """
    Validation finale du pipeline.
    """
    print("✅ Pipeline exécuté avec succès!")
    
    # Récupérer les résultats des tâches précédentes
    ti = context['ti']
    train_result = ti.xcom_pull(task_ids='train_model')
    
    print(f"🎯 Meilleure accuracy: {train_result.get('best_accuracy', 'N/A')}")
    
    return {'pipeline_status': 'success'}


# Création du DAG
with DAG(
    dag_id='news_classifier_pipeline',
    default_args=default_args,
    description='Pipeline complet de classification d\'articles d\'actualité',
    schedule_interval='@weekly',  # Exécution hebdomadaire
    catchup=False,
    tags=['nlp', 'classification', 'ml']
) as dag:
    
    # Tâches
    start = EmptyOperator(task_id='start')
    
    load_data = PythonOperator(
        task_id='load_data',
        python_callable=task_load_data,
        doc_md="JIRA-001: Chargement des données depuis Hugging Face"
    )
    
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=task_preprocess_data,
        doc_md="JIRA-002: Prétraitement des textes"
    )
    
    generate_embeddings = PythonOperator(
        task_id='generate_embeddings',
        python_callable=task_generate_embeddings,
        doc_md="JIRA-004: Génération des embeddings"
    )
    
    store_chromadb = PythonOperator(
        task_id='store_chromadb',
        python_callable=task_store_chromadb,
        doc_md="JIRA-005: Stockage dans ChromaDB"
    )
    
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=task_train_model,
        doc_md="JIRA-006: Entraînement des modèles"
    )
    
    validate = PythonOperator(
        task_id='validate_pipeline',
        python_callable=task_validate_pipeline,
        doc_md="Validation finale du pipeline"
    )
    
    end = EmptyOperator(task_id='end')
    
    # Définition du flux
    start >> load_data >> preprocess >> generate_embeddings >> store_chromadb >> train_model >> validate >> end
