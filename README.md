# Association Rule Classification

Association rule classification using Apriori algorithm on cardiovascular disease dataset. Compares ACR against Neural Network, Logistic Regression, and SVM.

## Dataset

Cardiovascular Disease Dataset - 70,000 patients
Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Features: age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity
Target: cardiovascular disease presence (binary)

## Usage

Install dependencies:
```bash
pip install scikit-learn numpy pandas seaborn matplotlib
```

Run notebook:
```bash
jupyter notebook associative-rule-classification.ipynb
```

The notebook extracts data.zip, trains models, and compares results.
