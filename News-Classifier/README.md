# News Classifier

This project is an intelligent system that automatically classifies news articles into four categories: World, Sports, Business, and Sci/Tech. It utilizes a complete Natural Language Processing (NLP) pipeline, leveraging data from Hugging Face and employing various techniques for data loading, preprocessing, exploratory data analysis, and classification.

## Project Structure

```
News-Classifier
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── eda
│   │   ├── __init__.py
│   │   └── analysis.py
│   ├── models
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   └── nlp_pipeline.py
│   └── utils
│       ├── __init__.py
│       └── helpers.py
├── notebooks
│   └── exploratory_analysis.ipynb
├── tests
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   └── test_classifier.py
├── configs
│   └── config.yaml
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd News-Classifier
pip install -r requirements.txt
```

## Usage

1. **Data Loading**: The dataset is loaded from Hugging Face using the `load_data` function in `src/data/loader.py`.
2. **Text Preprocessing**: Text data is preprocessed using the `TextPreprocessor` class in `src/data/preprocessor.py`.
3. **Exploratory Data Analysis**: Perform EDA using the `EDA` class in `src/eda/analysis.py`.
4. **Model Training and Prediction**: Train the classification model using the `NewsClassifier` class in `src/models/classifier.py`.
5. **NLP Pipeline**: The entire process is orchestrated through the `NLPipeline` class in `src/pipeline/nlp_pipeline.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.