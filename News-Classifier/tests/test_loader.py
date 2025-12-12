import unittest
import pandas as pd
from src.data.loader import load_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        # Load the dataset
        df = load_data()
        
        # Check if the DataFrame is not empty
        self.assertFalse(df.empty, "DataFrame should not be empty")
        
        # Check if the DataFrame has the expected columns
        expected_columns = ['title', 'content', 'category']
        for column in expected_columns:
            self.assertIn(column, df.columns, f"Column '{column}' is missing from the DataFrame")
        
        # Check if the categories are within the expected set
        expected_categories = {'World', 'Sports', 'Business', 'Sci/Tech'}
        unique_categories = set(df['category'])
        self.assertTrue(unique_categories.issubset(expected_categories), "Categories in DataFrame are not as expected")

if __name__ == '__main__':
    unittest.main()