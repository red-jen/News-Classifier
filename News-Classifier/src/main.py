import pandas as pd
from src.data.loader import load_data
from src.data.preprocessor import TextPreprocessor
from src.eda.analysis import EDA
from src.pipeline.nlp_pipeline import NLPipeline
from src.models.classifier import NewsClassifier

def main():
    # Load the dataset
    df = load_data()

    # Initialize the text preprocessor
    preprocessor = TextPreprocessor()
    df = preprocessor.normalize_text(df)

    # Perform exploratory data analysis
    eda = EDA(df)
    eda.visualize_data_distribution()
    eda.generate_summary_statistics()

    # Initialize and run the NLP pipeline
    nlp_pipeline = NLPipeline(df)
    nlp_pipeline.run()

    # Initialize and train the classifier
    classifier = NewsClassifier()
    classifier.train(df)

    # Example prediction
    sample_text = "The stock market is experiencing unprecedented growth."
    prediction = classifier.predict(sample_text)
    print(f"Predicted category: {prediction}")

if __name__ == "__main__":
    main()