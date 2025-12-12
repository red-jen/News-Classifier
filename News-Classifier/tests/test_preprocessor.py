import unittest
import pandas as pd
from src.data.preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.sample_data = pd.Series([
            "This is a sample news article about sports.",
            "Another article focusing on business and economy.",
            "Latest updates in science and technology."
        ])

    def test_normalization(self):
        normalized_data = self.preprocessor.normalize(self.sample_data)
        expected_data = pd.Series([
            "this is a sample news article about sports",
            "another article focusing on business and economy",
            "latest updates in science and technology"
        ])
        pd.testing.assert_series_equal(normalized_data, expected_data)

    def test_remove_duplicates(self):
        data_with_duplicates = pd.Series([
            "This is a news article.",
            "This is a news article.",
            "This is another article."
        ])
        unique_data = self.preprocessor.remove_duplicates(data_with_duplicates)
        expected_data = pd.Series([
            "This is a news article.",
            "This is another article."
        ])
        pd.testing.assert_series_equal(unique_data.reset_index(drop=True), expected_data.reset_index(drop=True))

    def test_remove_stopwords(self):
        data_with_stopwords = pd.Series([
            "This is a sample news article.",
            "Another article focusing on business."
        ])
        cleaned_data = self.preprocessor.remove_stopwords(data_with_stopwords)
        expected_data = pd.Series([
            "sample news article.",
            "Another article focusing business."
        ])
        pd.testing.assert_series_equal(cleaned_data, expected_data)

    def test_remove_punctuation(self):
        data_with_punctuation = pd.Series([
            "Hello, world!",
            "Python is great; isn't it?"
        ])
        cleaned_data = self.preprocessor.remove_punctuation(data_with_punctuation)
        expected_data = pd.Series([
            "Hello world",
            "Python is great isnt it"
        ])
        pd.testing.assert_series_equal(cleaned_data, expected_data)

if __name__ == '__main__':
    unittest.main()