import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------
# 1. Load or create dataset
# -------------------------
data_path = "data/data.csv"
if not os.path.exists("data"):
    os.makedirs("data")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    print("No data/data.csv found — creating a small synthetic demo dataset.\n")
    df = pd.DataFrame({
        "age": [25, 35, np.nan, 45, 23, 40, np.nan, 30, 50, 28],
        "salary": [40000, 80000, 50000, 120000, 35000, 60000, 70000, 90000, 110000, np.nan],
        "department": ["sales", "engineering", "sales", "management", "hr", "engineering",
                       "hr", "sales", np.nan, "management"],
        "tenure": [1, 8, 3, 15, 0.5, 6, 10, 4, 20, 2],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    })
    df.to_csv(data_path, index=False)

print("Columns in dataset:", df.columns.tolist())
print(df.head(), "\n")

# -------------------------
# 2. Handle missing values
# -------------------------
print("Missing value counts:\n", df.isnull().sum(), "\n")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")

# -------------------------
# 3. Define pipeline
# -------------------------
numeric_features = ["age", "salary", "tenure"]
categorical_features = ["department"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Final pipeline
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# -------------------------
# 4. Train & Evaluate
# -------------------------
clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print("\nAccuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# -------------------------
# 5. Save pipeline
# -------------------------
artifact_dir = "artifacts"
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

artifact_path = os.path.join(artifact_dir, "pipeline.pkl")
joblib.dump(clf_pipeline, artifact_path)

print(f"\n✅ Pipeline saved to {artifact_path}")