import unittest
from src.models.classifier import NewsClassifier

class TestNewsClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = NewsClassifier()

    def test_train(self):
        # Assuming we have a method to generate dummy training data
        train_data, train_labels = self.generate_dummy_data()
        self.classifier.train(train_data, train_labels)
        self.assertIsNotNone(self.classifier.model)

    def test_predict(self):
        # Assuming we have a method to generate dummy test data
        test_data = self.generate_dummy_test_data()
        predictions = self.classifier.predict(test_data)
        self.assertEqual(len(predictions), len(test_data))

    def generate_dummy_data(self):
        # Dummy data for testing
        data = ["This is a world news article.", "This is a sports article."]
        labels = ["World", "Sports"]
        return data, labels

    def generate_dummy_test_data(self):
        # Dummy test data for prediction
        return ["This is a business news article.", "This is a sci-tech article."]

if __name__ == '__main__':
    unittest.main()