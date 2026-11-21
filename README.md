#  Heart Disease Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Academic%20Mini%20Project-success)

## 📌 Project Overview

**Heart Disease Prediction Using Machine Learning** is an academic mini project developed using Python and Scikit-learn.

The project aims to predict the presence or absence of heart disease based on selected patient-related attributes from a publicly available dataset.

It demonstrates a complete Machine Learning workflow, including:

- Dataset collection and understanding
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Train-test splitting
- Machine Learning model comparison
- Model evaluation
- Scikit-learn Pipeline
- 5-Fold Cross-Validation
- Hyperparameter tuning using GridSearchCV
- Model saving using Pickle
- Interactive Streamlit application
- Git and GitHub version control
- Cloud deployment

**Project Type:** Supervised Machine Learning – Binary Classification

**Programming Language:** Python

**Application Framework:** Streamlit

---

## 🎯 Project Objectives

1. Analyse a publicly available heart disease dataset.
2. Perform data cleaning and preprocessing.
3. Understand feature distributions through EDA.
4. Implement and compare three classification algorithms.
5. Evaluate model performance using classification metrics.
6. Develop a reproducible Scikit-learn Pipeline.
7. Apply five-fold cross-validation.
8. Optimize model hyperparameters using GridSearchCV.
9. Save the final fitted model using Pickle.
10. Develop an interactive prediction application using Streamlit.
11. Maintain the project using Git and GitHub.
12. Deploy the application online.

---

## 📊 Dataset Information

**Dataset Name:** Heart Disease Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:**  
https://archive.ics.uci.edu/dataset/45/heart+disease

**Selected Dataset:** Cleveland Heart Disease Dataset

| Attribute | Information |
|---|---|
| Total records | 303 |
| Predictive features | 13 |
| Target variable | num |
| Machine Learning task | Binary Classification |
| Data type | Numerical and Categorical |

### Dataset Features

| Feature | Description |
|---|---|
| age | Patient age |
| sex | Sex indicator |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels |
| thal | Thalassemia category |
| num | Target variable |

### Target Variable

The original target variable contains values from 0 to 4.

For binary classification:

- `0` = No heart disease indicated
- `1, 2, 3, 4` = Heart disease indicated

The target is converted into two classes during preprocessing.

---

## 🤖 Machine Learning Algorithms

The project implements and compares the following algorithms:

### 1. Logistic Regression
A supervised classification algorithm used as a baseline model.

### 2. Decision Tree Classifier
A tree-based algorithm that makes predictions through a sequence of decision rules.

### 3. Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to improve predictive performance.

### Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (where appropriate)

The final algorithm is selected based on actual experimental results and generalization performance.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Data analysis and experimentation |
| VS Code | Development environment |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Pickle | Model serialization |
| Streamlit | Web application |
| Git | Version control |
| GitHub | Repository hosting |
| Streamlit Community Cloud | Deployment |

---

## 📁 Project Structure

```text
Heart_Disease_Prediction/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   └── heart_disease_analysis.ipynb
│
├── models/
│   └── model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── report.pdf
```

### File Description

| File | Description |
|---|---|
| data/heart.csv | Dataset file |
| notebooks/heart_disease_analysis.ipynb | EDA and ML implementation |
| models/model.pkl | Saved fitted ML pipeline |
| app.py | Streamlit application |
| train_model.py | Model training script |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| .gitignore | Files excluded from Git |
| report.pdf | Final academic project report |

---

## ⚙️ Installation and Setup

### Step 1: Install Python

Download Python from:

https://www.python.org/downloads/

Python 3.10 or a compatible newer version is recommended, subject to library compatibility.

Verify installation:

```bash
python --version
```

### Step 2: Clone the GitHub Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd Heart_Disease_Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download the Dataset

1. Visit the UCI Heart Disease dataset.
2. Download the processed Cleveland dataset.
3. Place the CSV file inside the `data/` directory.
4. Name the file `heart.csv`.
5. Confirm that the target column is named `num`.

Follow the dataset documentation to ensure the correct processed Cleveland subset is used.

---

## 📈 Implementation Workflow

### Step 1: Data Understanding

- Load the dataset.
- Display the first five records.
- Inspect dataset dimensions.
- Check data types.
- Generate statistical summaries.
- Identify missing values and duplicates.

### Step 2: Exploratory Data Analysis

Create visualizations:

- Target distribution.
- Age distribution.
- Cholesterol distribution.
- Blood pressure distribution.
- Correlation heatmap.
- Chest pain distribution.
- Maximum heart rate distribution.

### Step 3: Data Preprocessing

- Handle missing values.
- Examine duplicate observations.
- Separate features and target.
- Convert the target into binary classification.
- Encode categorical features if required.
- Scale numerical features where appropriate.

### Step 4: Train-Test Split

Split the dataset into:

- 80% Training data
- 20% Testing data

Use stratified splitting to preserve class proportions.

### Step 5: Model Training

Train and compare:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Step 6: Model Evaluation

Calculate:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

### Step 7: Pipeline Development

Combine preprocessing and model training using Scikit-learn Pipeline.

### Step 8: Cross-Validation

Apply StratifiedKFold with five folds to evaluate model performance on the training dataset.

### Step 9: Hyperparameter Tuning

Use GridSearchCV to identify suitable model parameters.

### Step 10: Save the Final Model

Save the complete fitted pipeline:

```python
import pickle

with open("models/model.pkl", "wb") as file:
    pickle.dump(best_pipeline, file)
```

### Step 11: Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌐 Streamlit Application

The project includes an interactive web application.

### Application Features

- Simple and user-friendly interface.
- Patient attribute input fields.
- Input validation.
- Predict button.
- Saved model loading.
- Machine Learning prediction output.
- Medical disclaimer.

### Application Workflow

```text
User Input
    |
    v
Input Validation
    |
    v
Load Saved Pipeline
    |
    v
Preprocessing
    |
    v
ML Classification
    |
    v
Display Prediction
```

---

## 📊 Experimental Results

Complete this table after executing the machine-learning notebook.

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | Pending | Pending | Pending | Pending |
| Decision Tree | Pending | Pending | Pending | Pending |
| Random Forest | Pending | Pending | Pending | Pending |

**Best Model:** To be updated after experimentation.

**Best Cross-Validation Score:** To be updated.

**Best Hyperparameters:** To be updated.

No performance values are assumed or fabricated.

---

## 🚀 Deployment

The application can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Upload the completed project to GitHub.
2. Ensure `app.py`, `requirements.txt` and `models/model.pkl` are available.
3. Visit https://share.streamlit.io/
4. Sign in using GitHub.
5. Select Create App.
6. Choose your repository.
7. Select the `main` branch.
8. Set the entry-point file to `app.py`.
9. Click Deploy.
10. Test the application using different valid inputs.

Deployment documentation:

https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy

### Project Links

**GitHub Repository:**  
[Add your GitHub repository URL]

**Live Application:**  
[Add your deployed Streamlit URL]

---

## 👥 Team Members and Contributions

**Team Size:** 3 Members

### Member 1 – Dataset and EDA
- Problem definition
- Dataset collection
- Data understanding
- Data cleaning
- Exploratory Data Analysis
- Data visualization

### Member 2 – Machine Learning
- Logistic Regression implementation
- Decision Tree implementation
- Random Forest implementation
- Model comparison
- Model evaluation

### Member 3 – Optimization and Deployment
- Pipeline development
- Cross-validation
- GridSearchCV
- Pickle model saving
- Streamlit development
- GitHub management
- Application deployment

All members are responsible for understanding the complete project workflow.

---

## ⚠️ Limitations and Disclaimer

This project is developed strictly for educational and academic purposes.

- The dataset contains a limited number of records.
- The dataset may not represent all populations.
- Model predictions may contain errors.
- The model has not undergone clinical validation.
- Predictions must not be used to diagnose, treat or rule out heart disease.
- Consult qualified healthcare professionals for medical concerns.

**This application is an academic machine-learning prototype and is not a medical diagnostic system.**

---

## 🔮 Future Enhancements

1. Use larger and more diverse datasets.
2. Explore XGBoost and other ensemble algorithms.
3. Integrate SHAP for model explainability.
4. Improve class imbalance handling.
5. Perform independent external validation.
6. Improve application design and accessibility.
7. Conduct clinical validation before considering any real-world healthcare application.

---

## 📚 References

1. Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology, 64*(5), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9

2. UCI Machine Learning Repository – Heart Disease Dataset. https://archive.ics.uci.edu/dataset/45/heart+disease

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

4. Python Documentation – https://docs.python.org/3/

5. Scikit-learn Documentation – https://scikit-learn.org/stable/

6. Streamlit Documentation – https://docs.streamlit.io/

7. Git Documentation – https://git-scm.com/doc

8. GitHub Documentation – https://docs.github.com/

---

## ⭐ Acknowledgement

We express our sincere gratitude to our faculty members and the Faculty of Computer Science at Lachoo Memorial College of Science & Technology (Autonomous), Jodhpur, for their guidance and support in completing this Machine Learning Mini Project.

We also acknowledge the UCI Machine Learning Repository for providing the publicly available dataset used in this academic project.

---

**Developed as part of the MCA Machine Learning Mini Project – Academic Year 2026–2027.**
