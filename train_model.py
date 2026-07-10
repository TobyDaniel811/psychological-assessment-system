"""
train_model.py
----------------------------------------------------------------------
Web-Based Behavioural Survey and Psychological Assessment System
Random Forest training pipeline.

This script:
  1. Loads the real dataset (Psychological_Assessment_Dataset.xlsx)
  2. Cleans duplicate rows
  3. Encodes ordinal features (frequency/intensity columns) as 0,1,2,3...
  4. One-hot encodes nominal features (category columns with no order)
  5. Extracts a clean target label from "Condition Summary"
  6. Splits data 80/20 with stratification
  7. Trains a Random Forest with class_weight='balanced'
  8. Evaluates with accuracy, precision, recall, F1-score, confusion matrix
  9. Saves the trained model + the list of expected input columns with joblib

Run with:   python train_model.py
----------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")          # no GUI needed, just save the image
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# STEP 1: Load the dataset
# ----------------------------------------------------------------------
DATA_PATH = "dataset/Psychological_Assessment_Dataset.xlsx"

print("STEP 1: Loading dataset...")
df = pd.read_excel(DATA_PATH)
print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns.\n")

# ----------------------------------------------------------------------
# STEP 2: Remove duplicate rows
# ----------------------------------------------------------------------
print("STEP 2: Removing duplicate rows...")
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
print(f"  Removed {before - after} duplicate row(s). {after} rows remain.\n")

# ----------------------------------------------------------------------
# STEP 3: Identify columns
# ----------------------------------------------------------------------
TARGET_COL = "Condition Summary"

# These columns have a natural ORDER (frequency / intensity), so they
# are encoded as ordinal integers 0, 1, 2, 3... in increasing severity.
ORDINAL_MAPS = {
    "Anxious Social Scale: On a scale of 1-10, how often have you felt anxious in social situations recently?": {
        "Not at all": 0, "Rarely anxious": 1, "Slightly anxious": 2,
        "Mildly anxious": 3, "Somewhat anxious": 4, "Moderately anxious": 5,
        "Fairly anxious": 6, "Very anxious": 7, "Extremely anxious": 8,
        "Constantly anxious": 9,
    },
    "Lack of Interest: How often have you felt a lack of interest or pleasure in daily activities?": {
        "Never": 0, "Rarely": 1, "Occasionally": 2, "Frequently": 3, "Always": 4,
    },
    "Enjoyable Activities: How often do you engage in activities you enjoy or that help you relax?": {
        "Never": 0, "Rarely": 1, "Once a week": 2, "A few times a week": 3, "Daily": 4,
    },
    "Physical Anxiety Symptoms: Have you had any physical symptoms of anxiety (e.g., heart palpitations, sweating, shortness of breath)?": {
        "No, not at all": 0, "Rarely": 1, "Yes, occasionally": 2, "Yes, frequently": 3,
    },
    "Concentration Difficulty: How often do you find it difficult to concentrate on tasks?": {
        "Never": 0, "Occasionally": 1, "Frequently": 2, "Constantly": 3,
    },
}

# These columns have NO natural order, so they will be one-hot encoded.
NOMINAL_COLS = [
    "Mood: How would you describe your mood over the past two weeks?",
    "Anxiety Triggers: Have you experienced any of the following anxiety triggers in the past month?",
    "Sleep Quality: How would you rate the quality of your sleep over the past week?",
    "Appetite Change: Have you noticed any significant changes in your appetite?",
    "Coping Strategies: What coping strategies have you used when feeling stressed or anxious?",
]

# ----------------------------------------------------------------------
# STEP 4: Encode ordinal columns
# ----------------------------------------------------------------------
print("STEP 3: Encoding ordinal (ordered) features...")
for col, mapping in ORDINAL_MAPS.items():
    short_name = col.split(":")[0]
    df[col] = df[col].map(mapping)
    print(f"  Encoded '{short_name}' -> integers 0-{len(mapping)-1}")
print()

# ----------------------------------------------------------------------
# STEP 5: One-hot encode nominal columns
# ----------------------------------------------------------------------
print("STEP 4: One-hot encoding nominal (category) features...")
for col in NOMINAL_COLS:
    short_name = col.split(":")[0]
    n_categories = df[col].nunique()
    print(f"  One-hot encoding '{short_name}' -> {n_categories} new columns")

df_encoded = pd.get_dummies(df, columns=NOMINAL_COLS)
print(f"\n  Dataset shape after one-hot encoding: {df_encoded.shape}\n")

# ----------------------------------------------------------------------
# STEP 6: Prepare the clean target label
# ----------------------------------------------------------------------
print("STEP 5: Preparing the target variable...")
df_encoded["Condition"] = df_encoded[TARGET_COL].str.split(":").str[0].str.strip()
print("  Extracted clean class names from 'Condition Summary'.")
print("  Class distribution:")
print(df_encoded["Condition"].value_counts().to_string())
print()

# ----------------------------------------------------------------------
# STEP 7: Build final X (features) and y (target)
# ----------------------------------------------------------------------
X = df_encoded.drop(columns=[TARGET_COL, "Condition"])
y = df_encoded["Condition"]

# Save the exact list+order of feature columns the model expects.
# Flask will need this later to build a matching input vector.
FEATURE_COLUMNS = list(X.columns)

print(f"STEP 6: Final feature matrix shape: {X.shape}")
print(f"  Number of input features after encoding: {X.shape[1]}")
print(f"  Number of target classes: {y.nunique()}\n")

# ----------------------------------------------------------------------
# STEP 8: Train/test split (80/20, stratified)
# ----------------------------------------------------------------------
print("STEP 7: Splitting into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # ensures rare classes (e.g. PTSD) appear proportionally
                         # in both the training set and the test set
)
print(f"  Training set: {X_train.shape[0]} rows")
print(f"  Test set:     {X_test.shape[0]} rows\n")

# ----------------------------------------------------------------------
# STEP 9: Train the Random Forest classifier
# ----------------------------------------------------------------------
print("STEP 8: Training the Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,          # reduced from 300 -- keeps file size deployable
    max_depth=20,               # caps tree depth -- prevents oversized trees
    class_weight="balanced",   # compensates for the severe class imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  Training complete.\n")

# ----------------------------------------------------------------------
# STEP 10: Evaluate the model
# ----------------------------------------------------------------------
print("STEP 9: Evaluating the model on the held-out test set...")
y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\n  Overall Accuracy : {acc:.4f}")
print(f"  Weighted Precision: {prec:.4f}")
print(f"  Weighted Recall   : {rec:.4f}")
print(f"  Weighted F1-score : {f1:.4f}\n")

print("  Full per-class classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ----------------------------------------------------------------------
# STEP 11: Confusion matrix
# ----------------------------------------------------------------------
print("STEP 10: Generating confusion matrix...")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=True)
plt.xlabel("Predicted Condition")
plt.ylabel("Actual Condition")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

os.makedirs("documentation", exist_ok=True)
plt.savefig("documentation/confusion_matrix.png", dpi=150)
print("  Confusion matrix image saved to documentation/confusion_matrix.png\n")

# ----------------------------------------------------------------------
# STEP 12: Save the trained model + feature column list
# ----------------------------------------------------------------------
print("STEP 11: Saving model artifacts...")
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/random_forest_model.pkl")
joblib.dump(FEATURE_COLUMNS, "model/feature_columns.pkl")
joblib.dump(ORDINAL_MAPS, "model/ordinal_maps.pkl")
joblib.dump(NOMINAL_COLS, "model/nominal_cols.pkl")

print("  Saved: model/random_forest_model.pkl")
print("  Saved: model/feature_columns.pkl   (column order Flask must match)")
print("  Saved: model/ordinal_maps.pkl      (encoding dictionaries for ordinal fields)")
print("  Saved: model/nominal_cols.pkl      (list of one-hot columns)")
print("\nTraining pipeline complete.")