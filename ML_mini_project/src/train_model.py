from pathlib import Path
import pickle
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 1. Download dataset
dataset = fetch_ucirepo(id=350)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

X.columns = [str(c).strip().upper() for c in X.columns]

if "ID" in X.columns:
    X = X.drop(columns=["ID"])

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = pd.to_numeric(y, errors="raise").astype(int)

columns = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

if len(X.columns) != len(columns):
    raise ValueError("Unexpected dataset feature count.")

X.columns = columns

df = X.copy()
df["default_payment_next_month"] = y.values
df.to_csv(DATA_DIR / "credit_card_default.csv", index=False)

# 2. Split data
X = df.drop(columns=["default_payment_next_month"])
y = df["default_payment_next_month"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Preprocessing
categorical = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

numerical = [c for c in X.columns if c not in categorical]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical),
    ("cat", categorical_pipeline, categorical)
])

# 4. Model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

# 5. Hyperparameter tuning
parameters = {
    "classifier__n_estimators": [150, 250],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    pipeline,
    parameters,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# 6. Final evaluation
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

print("Best parameters:", search.best_params_)
print("Cross-validation F1:", search.best_score_)
print("Test accuracy:", accuracy_score(y_test, predictions))
print("Test precision:", precision_score(y_test, predictions, zero_division=0))
print("Test recall:", recall_score(y_test, predictions, zero_division=0))
print("Test F1:", f1_score(y_test, predictions, zero_division=0))
print("Test ROC-AUC:", roc_auc_score(y_test, probabilities))

report = classification_report(
    y_test,
    predictions,
    target_names=["No Default", "Default"],
    zero_division=0
)

print(report)

# 7. Save model
model_path = MODEL_DIR / "model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(best_model, file)

# 8. Save report
with open(REPORT_DIR / "classification_report.txt", "w") as file:
    file.write(report)

print("Training completed.")
print("Saved model:", model_path)
