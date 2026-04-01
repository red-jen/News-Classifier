
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# Mapping des labels
LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']


class NewsClassifier:
    """
    Classificateur d'articles d'actualité basé sur les embeddings.
    
    Supporte plusieurs algorithmes ML:
    - Logistic Regression (par défaut)
    - Random Forest
    - SVM
    - KNN
    - MLP (Neural Network)
    """
    
    AVAILABLE_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'mlp': MLPClassifier
    }
    
    DEFAULT_PARAMS = {
        'logistic_regression': {'max_iter': 1000, 'random_state': 42, 'n_jobs': -1},
        'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
        'svm': {'kernel': 'rbf', 'random_state': 42, 'probability': True},
        'knn': {'n_neighbors': 5, 'n_jobs': -1},
        'mlp': {'hidden_layer_sizes': (256, 128), 'max_iter': 500, 'random_state': 42}
    }
    
    def __init__(self, model_type: str = 'logistic_regression', **kwargs):
        """
        Initialise le classificateur.
        
        Args:
            model_type: Type de modèle ('logistic_regression', 'random_forest', etc.)
            **kwargs: Paramètres additionnels pour le modèle
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modèle '{model_type}' non supporté. "
                           f"Choix: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_class = self.AVAILABLE_MODELS[model_type]
        
        # Fusionner les paramètres par défaut avec ceux fournis
        params = {**self.DEFAULT_PARAMS[model_type], **kwargs}
        self.model = self.model_class(**params)
        
        self.is_trained = False
        self.training_history: Dict[str, Any] = {}
        
        print(f"✅ Classificateur initialisé: {model_type}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Entraîne le modèle sur les embeddings.
        
        Args:
            X_train: Embeddings d'entraînement
            y_train: Labels d'entraînement
            
        Returns:
            Métriques d'entraînement
        """
        print(f"⏳ Entraînement du modèle {self.model_type}...")
        print(f"   Shape: X={X_train.shape}, y={y_train.shape}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Évaluation sur le train set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        self.training_history['train_accuracy'] = train_accuracy
        
        print(f"✅ Entraînement terminé! Accuracy train: {train_accuracy:.4f}")
        
        return {'train_accuracy': train_accuracy}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes pour les embeddings.
        
        Args:
            X: Embeddings à classifier
            
        Returns:
            Array des prédictions
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités de chaque classe.
        
        Args:
            X: Embeddings à classifier
            
        Returns:
            Array des probabilités
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné!")
        
        # SVC sans probability=True n'a pas predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback: utiliser les prédictions pour créer des probabilités simulées
            predictions = self.model.predict(X)
            n_classes = len(np.unique(predictions)) if len(predictions) > 0 else 4
            n_classes = max(n_classes, 4)  # Au moins 4 classes
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0  # 100% pour la classe prédite
            return proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Évalue le modèle sur le jeu de test.
        
        Args:
            X_test: Embeddings de test
            y_test: Labels de test
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        print(f"⏳ Évaluation du modèle...")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=LABEL_NAMES, output_dict=True
            )
        }
        
        self.training_history['test_metrics'] = metrics
        
        print(f"\n📊 Résultats d'évaluation:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall:    {metrics['recall_macro']:.4f}")
        print(f"   F1-Score:  {metrics['f1_macro']:.4f}")
        
        print(f"\n📋 Rapport de classification:")
        print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
        
        return metrics
    
    def check_overfitting(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Vérifie l'overfitting en comparant les performances train/test.
        
        Args:
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
            
        Returns:
            Analyse de l'overfitting
        """
        print("\n🔍 Analyse de l'overfitting...")
        
        # Accuracy sur train et test
        train_acc = accuracy_score(y_train, self.predict(X_train))
        test_acc = accuracy_score(y_test, self.predict(X_test))
        
        gap = train_acc - test_acc
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        analysis = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy_gap': gap,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'is_overfitting': gap > 0.05  # Seuil de 5%
        }
        
        print(f"\n📊 Analyse Overfitting:")
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy:  {test_acc:.4f}")
        print(f"   Gap:            {gap:.4f}")
        print(f"   CV Mean (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        if analysis['is_overfitting']:
            print(f"\n⚠️ ATTENTION: Possible overfitting détecté (gap > 5%)")
        else:
            print(f"\n✅ Pas d'overfitting significatif détecté")
            
        return analysis
    
    def save_model(self, filepath: str) -> None:
        """Sauvegarde le modèle entraîné."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné!")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'training_history': self.training_history
        }, filepath)
        
        print(f"✅ Modèle sauvegardé: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NewsClassifier':
        """Charge un modèle sauvegardé."""
        data = joblib.load(filepath)
        
        # Gérer les deux formats: dict (NewsClassifier) ou modèle sklearn brut
        if isinstance(data, dict) and 'model' in data:
            # Format NewsClassifier
            classifier = cls(model_type=data['model_type'])
            classifier.model = data['model']
            classifier.training_history = data.get('training_history', {})
        else:
            # Format sklearn brut (sauvegardé depuis le notebook)
            model_type = type(data).__name__.lower()
            if 'logistic' in model_type:
                model_type = 'logistic_regression'
            elif 'svc' in model_type or 'svm' in model_type:
                model_type = 'svm'
            elif 'randomforest' in model_type:
                model_type = 'random_forest'
            elif 'kneighbors' in model_type:
                model_type = 'knn'
            else:
                model_type = 'logistic_regression'
            
            classifier = cls(model_type=model_type)
            classifier.model = data
            classifier.training_history = {}
        
        classifier.is_trained = True
        print(f"✅ Modèle chargé: {filepath}")
        return classifier


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare plusieurs modèles de classification.
    
    Returns:
        DataFrame avec les résultats comparatifs
    """
    results = []
    
    models_to_test = ['logistic_regression', 'random_forest', 'svm', 'knn']
    
    for model_type in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_type}")
        print('='*50)
        
        classifier = NewsClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        overfitting = classifier.check_overfitting(X_train, y_train, X_test, y_test)
        
        results.append({
            'model': model_type,
            'train_accuracy': overfitting['train_accuracy'],
            'test_accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_macro'],
            'cv_mean': overfitting['cv_mean'],
            'overfitting_gap': overfitting['accuracy_gap']
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_accuracy', ascending=False)
    
    print("\n" + "="*60)
    print("📊 COMPARAISON DES MODÈLES")
    print("="*60)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    # Test du module
    print("Test du module classifier...")
    
    # Données synthétiques
    X_train = np.random.rand(1000, 384)
    y_train = np.random.randint(0, 4, 1000)
    X_test = np.random.rand(200, 384)
    y_test = np.random.randint(0, 4, 200)
    
    classifier = NewsClassifier('logistic_regression')
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)
    classifier.check_overfitting(X_train, y_train, X_test, y_test)
