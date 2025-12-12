"""
JIRA-007: Streamlit Application
================================
Application web pour la classification d'articles d'actualité.
Interface utilisateur intuitive avec prédiction en temps réel.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ajout du chemin src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.embeddings.generator import EmbeddingGenerator
from src.models.classifier import NewsClassifier
from src.data.preprocessor import TextPreprocessor


# Configuration de la page
st.set_page_config(
    page_title="📰 News Classifier",
    page_icon="📰",
    layout="wide"
)

# Constantes
LABEL_NAMES = {
    0: ('World', '🌍'),
    1: ('Sports', '⚽'),
    2: ('Business', '💼'),
    3: ('Sci/Tech', '🔬')
}

MODEL_PATH = "./models/best_model.joblib"


@st.cache_resource
def load_models():
    """Charge les modèles (embeddings et classificateur)."""
    embedding_generator = EmbeddingGenerator()
    embedding_generator.load_model()
    
    try:
        classifier = NewsClassifier.load_model(MODEL_PATH)
    except FileNotFoundError:
        st.warning("⚠️ Modèle non trouvé. Utilisation d'un modèle de démonstration.")
        classifier = None
    
    preprocessor = TextPreprocessor()
    
    return embedding_generator, classifier, preprocessor


def predict_category(text: str, embedding_generator, classifier, preprocessor):
    """Prédit la catégorie d'un article."""
    # Prétraitement
    clean_text = preprocessor.preprocess(text)
    
    # Génération de l'embedding
    embedding = embedding_generator.generate_embeddings([clean_text], show_progress=False)
    
    # Prédiction
    if classifier is not None:
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
    else:
        # Mode démo sans modèle
        prediction = np.random.randint(0, 4)
        probabilities = np.random.dirichlet(np.ones(4))
    
    return prediction, probabilities, clean_text


def main():
    """Application principale."""
    
    # En-tête
    st.title("📰 News Article Classifier")
    st.markdown("""
    ### Classification automatique d'articles d'actualité
    
    Cette application utilise le **Machine Learning** et les **embeddings Sentence Transformers** 
    pour classifier automatiquement les articles d'actualité en 4 catégories:
    
    | Catégorie | Description |
    |-----------|-------------|
    | 🌍 **World** | Actualités internationales |
    | ⚽ **Sports** | Événements sportifs |
    | 💼 **Business** | Économie et finance |
    | 🔬 **Sci/Tech** | Science et technologie |
    """)
    
    st.divider()
    
    # Chargement des modèles
    with st.spinner("Chargement des modèles..."):
        embedding_generator, classifier, preprocessor = load_models()
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Entrez votre article")
        
        # Zone de texte pour l'article
        article_text = st.text_area(
            "Collez ou tapez le texte de l'article ici:",
            height=200,
            placeholder="Example: The stock market reached new highs today as technology companies reported strong earnings..."
        )
        
        # Exemples pré-définis
        st.markdown("**Ou essayez un exemple:**")
        
        examples = {
            "🌍 World": "The United Nations held an emergency meeting to discuss the ongoing humanitarian crisis in the region. World leaders called for immediate action and increased aid.",
            "⚽ Sports": "The championship final was an incredible match! The home team scored in the last minute to win 3-2, securing their first title in 20 years.",
            "💼 Business": "Apple Inc. reported record quarterly earnings, surpassing Wall Street expectations. The company's stock rose 5% in after-hours trading.",
            "🔬 Sci/Tech": "Scientists have developed a new AI algorithm that can predict protein structures with unprecedented accuracy, potentially revolutionizing drug discovery."
        }
        
        example_cols = st.columns(4)
        for i, (cat, text) in enumerate(examples.items()):
            if example_cols[i].button(cat, use_container_width=True):
                article_text = text
                st.rerun()
    
    with col2:
        st.subheader("🎯 Résultat de la classification")
        
        if article_text and len(article_text) > 10:
            with st.spinner("Analyse en cours..."):
                prediction, probabilities, clean_text = predict_category(
                    article_text, embedding_generator, classifier, preprocessor
                )
            
            category_name, emoji = LABEL_NAMES[prediction]
            
            # Affichage du résultat principal
            st.success(f"### {emoji} {category_name}")
            st.metric("Confiance", f"{probabilities[prediction]*100:.1f}%")
            
            # Probabilités pour toutes les catégories
            st.markdown("**Probabilités par catégorie:**")
            
            prob_df = pd.DataFrame({
                'Catégorie': [f"{LABEL_NAMES[i][1]} {LABEL_NAMES[i][0]}" for i in range(4)],
                'Probabilité': probabilities * 100
            })
            
            st.bar_chart(prob_df.set_index('Catégorie'))
            
            # Texte prétraité (expandable)
            with st.expander("Voir le texte prétraité"):
                st.text(clean_text[:500] + "..." if len(clean_text) > 500 else clean_text)
        else:
            st.info("👆 Entrez un article pour voir la prédiction")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **🛠️ Technologies utilisées:**
    - Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
    - Scikit-learn
    - ChromaDB
    - Streamlit
    
    *Projet réalisé dans le cadre d'une mission de classification automatique d'actualités.*
    """)


if __name__ == "__main__":
    main()
