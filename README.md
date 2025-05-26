# 🗳️ US Election Sentiment Predictor

A machine learning project analyzing public sentiment from social media to predict outcomes of U.S. presidential elections.

---

## 📊 Overview

This project leverages sentiment analysis on social media data to forecast U.S. presidential election results. It utilizes various machine learning models to interpret public opinion and predict electoral outcomes.

---

## 🧰 Features

- **Data Collection**: Scripts to download and preprocess social media data.
- **Sentiment Analysis**: Analyzes textual data to determine public sentiment.
- **Machine Learning Models**: Implements K-Nearest Neighbors, Linear Regression, and Neural Networks for prediction.
- **Performance Evaluation**: Assesses model accuracy and error metrics.

---

## 🗂️ Repository Structure

- `Team11_Data_Preprocessing.ipynb` – Data cleaning and preprocessing steps.
- `Team11_Downloader.py` – Script for downloading the necessary NLP features.
- `Team11_KNNclassifiermodel.ipynb` – KNN model implementation.
- `Team11_LinearRegressionModel.py` – Linear Regression model script.
- `Team11_NeuralNetworkClassification.ipynb` – Neural Network classifier notebook.
- `Team11_NeuralNetworkModelRegression.ipynb` – Neural Network regression model.
- `Team11_Polynomial.ipynb` - Polynomial Regression notebook.
- `Team11_PerformanceAnalyser.py` – Script to evaluate model performance.
- `*.csv` files – Datasets and model evaluation metrics.

---

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Zorojuro-des/US-Election-Sentiment-Predictor.git
   cd US-Election-Sentiment-Predictor
   ```
2. **Install Dependancies**:
    ```bash
    pip install -r requirements.txt
    ```
## 🚀 Usage
Execute the desired model script or notebook to train and evaluate predictions:
  ```bash
  python Team11_LinearRegressionModel.py
  ```
## 📈 Results

Model performance metrics are stored in corresponding .csv files:
Team11_KNNClassifierPerformanceMetrics.csv
Team11_LinearRegressionErrors.csv
Team11_NNClassification_EvaluationMetrics.csv
Team11_NeuralNetworkRegressionErrors.csv
Team11_PolynomialErrors.csv
These files contain accuracy scores, error rates, and other relevant evaluation metrics.

