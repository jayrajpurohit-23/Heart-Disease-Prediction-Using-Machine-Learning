#  Heart Disease Prediction Using Machine Learning

A Machine Learning-based classification project that predicts the possibility of heart disease using patient medical attributes. This project was developed as part of an MCA academic mini-project.

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Early identification of potential risk factors can help people seek timely medical attention.

This project uses machine learning classification algorithms to predict whether a person has an indication of heart disease based on medical attributes.

The project uses the **UCI Heart Disease Dataset (Cleveland subset)** and compares multiple machine learning algorithms to identify the best-performing model.

> **Disclaimer:** This project is developed for academic and educational purposes only. It is not a medical diagnostic tool and must not be used as a substitute for professional medical advice.

## 🎯 Objectives

- Analyze the Heart Disease dataset.
- Perform data cleaning and preprocessing.
- Explore the dataset using data visualization.
- Train and compare multiple machine learning algorithms.
- Apply 5-fold cross-validation.
- Perform hyperparameter tuning using GridSearchCV.
- Evaluate the final model using classification metrics.
- Save the trained machine learning pipeline.
- Develop a user-friendly prediction interface using Streamlit.

## 📊 Dataset Information

**Dataset:** UCI Heart Disease Dataset – Cleveland subset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

| Dataset Information | Details |
|---|---|
| Total Records | 303 |
| Input Features | 13 |
| Target Variable | Heart Disease |
| Class 0 | No Heart Disease Indicated |
| Class 1 | Heart Disease Indicated |
| Missing Values | 6 |
| Duplicate Records | 0 |

The original target variable contains values from 0 to 4. It was converted into binary classification:

- `0` → No Heart Disease Indicated
- `1–4` → Heart Disease Indicated

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

## 🤖 Machine Learning Algorithms

Four classification algorithms were compared:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classifier (SVC)
4. K-Nearest Neighbors (KNN)

**Best Selected Model:** Logistic Regression

The model was selected based on its mean cross-validation ROC-AUC score.

## ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Missing value handling using SimpleImputer.
- Median imputation for numerical features.
- Most frequent imputation for categorical features.
- StandardScaler for numerical feature scaling.
- OneHotEncoder for categorical feature encoding.
- ColumnTransformer for combining preprocessing steps.
- Pipeline for integrating preprocessing and model training.

## 📈 Model Evaluation

The dataset was divided using stratified train-test splitting:

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Cross-validation: 5-fold StratifiedKFold

### Algorithm Comparison

| Algorithm | Mean CV Accuracy | Mean CV ROC-AUC |
|---|---:|---:|
| Logistic Regression | 85.52% | 90.25% |
| Random Forest | 82.23% | 89.15% |
| SVC | 82.21% | 88.76% |
| KNN | 82.22% | 87.12% |

### Final Model Performance

After hyperparameter tuning using GridSearchCV, the final Logistic Regression model achieved the following test results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 88.52% |
| Precision | 83.87% |
| Recall | 92.86% |
| F1-Score | 88.14% |
| ROC-AUC | 96.65% |

**Confusion Matrix:**

| | Predicted: No Disease | Predicted: Disease |
|---|---:|---:|
| Actual: No Disease | 28 | 5 |
| Actual: Disease | 2 | 26 |

The final model correctly classified 54 out of 61 test records.

## 🔍 Hyperparameter Tuning

GridSearchCV was applied to Logistic Regression.

Parameters tested:

- `C`: [0.01, 0.1, 1, 10]
- `solver`: [lbfgs, liblinear]

Best parameters:

```python
{
    'classifier__C': 0.1,
    'classifier__solver': 'liblinear'
}
```

Best mean cross-validation ROC-AUC: **90.31%**

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb
│
├── models/
│   └── heart_disease_pipeline.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

*Update the file names and folders according to your actual GitHub repository.*

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Open the Project Folder

```bash
cd Heart-Disease-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook inside the `notebooks` folder and execute the cells in order.

### 6. Run the Streamlit Application

If `app.py` and the saved model are available:

```bash
streamlit run app.py
```

## 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
ucimlrepo
jupyter
notebook
streamlit
```


## 📚 References

1. UCI Machine Learning Repository – Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
2. Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology*.
3. Scikit-learn Documentation: https://scikit-learn.org/
4. Pandas Documentation: https://pandas.pydata.org/docs/
5. Streamlit Documentation: https://docs.streamlit.io/

**Dataset DOI:** https://doi.org/10.24432/C52P4X

## 📜 License

This project is intended for educational and academic purposes. You may add an MIT License if you want to allow others to reuse and modify the project under its terms.

---

**Developed as an MCA Machine Learning Mini Project**

**GitHub:** Jay Rajpurohit
