# Optimized-Machine-Learning-Models-for-Predictive-Insights-and-Performance-Analysis

This repository contains a comprehensive Jupyter Notebook implementation of various machine learning classification models.
---

## Project Overview

- **Notebook**: `MachineLearning.ipynb`

---

## Objective

To explore, implement, and compare the performance of multiple supervised learning algorithms on a given classification dataset. The notebook includes full preprocessing, model training, evaluation, and visualizations.

---

## Machine Learning Models Used

The following classifiers are implemented and compared:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Gradient Boosting**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Polynomial Regression**
- **CountVectorizer**
- **TfidfVectorizer**

---

## Workflow Overview

1. **Data Loading & Exploration**
   - Shape, summary statistics, null value checks
   - Class distribution analysis

2. **Preprocessing**
   - Encoding categorical variables
   - Handling missing data
   - Train-test split
   - Standardization / normalization

3. **Model Training & Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Cross-validation
   - ROC Curve / AUC (if applicable)
   - Silhouette Score

4. **Model Comparison**
   - Tabular comparison of performance metrics

---

## Key Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross-validated Score
- ROC AUC (if binary classification)
- Silhouette Score

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
```

---

## Getting Started

### Setup Environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run Notebook

```bash
jupyter notebook MachineLearning_PranavSunilRaja.ipynb
```

---

## Author

- **Pranav Sunil Raja**  
- Newcastle University  
- GitHub: [@pranavsraja](https://github.com/pranavsraja)

---

## 📎 License

This repository is intended for academic and educational purposes only.
