Dans un contexte de forte croissance du volume d’informations numériques, les plateformes de médias font face à un défi majeur : organiser, analyser et exploiter efficacement les flux massifs d’articles publiés chaque jour. En tant que développeur en intelligence artificielle au sein d’une entreprise spécialisée dans les services d’information digitale, votre mission consiste à concevoir un système intelligent capable de classer automatiquement les articles d’actualité en quatre catégories stratégiques : World, Sports, Business et Sci/Tech.

Ce projet repose sur la mise en place d’une pipeline NLP complète, intégrant les étapes suivantes :

Chargement des données depuis la plateforme Hugging Face, à partir du dataset SetFit/ag_news, en utilisant la bibliothèque datasets.
Conversion des données en DataFrame pandas.
Réalisation d’une analyse exploratoire des données (EDA).
Prétraitement des textes : Normalisation des textes, Suppression des doublons, Suppression des stopwords, Suppression de la ponctuation à l’aide des expressions régulières (regex).
Génération des embeddings à l’aide de Sentence Transformers, en utilisant le modèle paraphrase-multilingual-MiniLM-L12-v2 (ou un autre modèle de Hugging Face).
Sauvegarde des métadonnées (label et identifiant de chaque article).
Stockage des vecteurs d’embeddings avec leurs métadonnées et identifiants dans la base de données vectorielle ChromaDB : Une collection pour les données d’entraînement, Une collection pour les données de test.
Récupération des embeddings et entraînement des modèles de Machine Learning sur les vecteurs.
Évaluation des modèles et vérification de l’overfitting.
Intégration du modèle final dans une application Streamlit.
Orchestration complète de l’ensemble du pipeline avec Apache Airflow.