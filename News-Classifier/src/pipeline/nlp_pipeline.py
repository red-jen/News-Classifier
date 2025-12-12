from datasets import load_dataset
import pandas as pd
from src.data.preprocessor import TextPreprocessor
from src.eda.analysis import EDA
from src.models.classifier import NewsClassifier

class NLPipeline:
    def __init__(self):
        self.data = None
        self.preprocessor = TextPreprocessor()
        self.eda = EDA()
        self.classifier = NewsClassifier()

    def load_data(self, dataset_name):
        dataset = load_dataset(dataset_name)
        self.data = pd.DataFrame(dataset['train'])

    def preprocess_data(self):
        self.data['text'] = self.preprocessor.normalize(self.data['text'])
        self.data = self.preprocessor.remove_duplicates(self.data)
        self.data['text'] = self.preprocessor.remove_stopwords(self.data['text'])
        self.data['text'] = self.preprocessor.remove_punctuation(self.data['text'])

    def perform_eda(self):
        self.eda.visualize_data_distribution(self.data)
        self.eda.generate_summary_statistics(self.data)

    def train_model(self):
        self.classifier.train(self.data['text'], self.data['label'])

    def predict(self, new_data):
        return self.classifier.predict(new_data)