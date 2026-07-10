"""
app/predictor.py
----------------------------------------------------------------------
Loads the trained Random Forest model and the encoding artifacts saved
by train_model.py, and exposes one function:

    predict_condition(answers: dict) -> dict

`answers` must be a plain dictionary mapping each of the 10 original
question column names to the user's selected text answer, e.g.:

    {
      "Mood: How would you describe your mood over the past two weeks?": "Happiness",
      "Anxious Social Scale: ...": "Mildly anxious",
      ...
    }

This module guarantees the feature vector handed to the model has the
exact same columns, in the exact same order, as during training.
----------------------------------------------------------------------
"""

import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Loaded once when the Flask app starts -- not on every request.
model            = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
feature_columns  = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
ordinal_maps     = joblib.load(os.path.join(MODEL_DIR, "ordinal_maps.pkl"))
nominal_cols     = joblib.load(os.path.join(MODEL_DIR, "nominal_cols.pkl"))


def build_feature_vector(answers: dict) -> pd.DataFrame:
    """
    Converts a dict of {question_column: text_answer} into a single-row
    DataFrame with exactly the 33 columns the model was trained on,
    in the correct order, filled with 0/1 or ordinal integers.
    """
    # Start every one-hot column at 0 (False)
    row = {col: 0 for col in feature_columns}

    # --- Ordinal columns: look up the text answer, write the integer ---
    for col, mapping in ordinal_maps.items():
        if col in answers:
            text_value = answers[col]
            if text_value not in mapping:
                raise ValueError(
                    f"Unexpected value '{text_value}' for '{col.split(':')[0]}'. "
                    f"Valid options are: {list(mapping.keys())}"
                )
            row[col] = mapping[text_value]
        else:
            raise ValueError(f"Missing required answer for: {col.split(':')[0]}")

    # --- Nominal columns: turn on the matching one-hot column ---
    for col in nominal_cols:
        if col in answers:
            text_value = answers[col]
            one_hot_col = f"{col}_{text_value}"
            if one_hot_col not in row:
                raise ValueError(
                    f"Unexpected value '{text_value}' for '{col.split(':')[0]}'. "
                    f"This option was not seen during training."
                )
            row[one_hot_col] = 1
        else:
            raise ValueError(f"Missing required answer for: {col.split(':')[0]}")

    # Build a one-row DataFrame, with columns in the EXACT saved order.
    df_row = pd.DataFrame([row], columns=feature_columns)
    return df_row


def predict_condition(answers: dict) -> dict:
    """
    Takes the raw survey answers and returns the model's prediction
    plus a confidence score and the full probability breakdown.
    """
    X = build_feature_vector(answers)

    predicted_class = model.predict(X)[0]
    probabilities   = model.predict_proba(X)[0]
    class_names     = model.classes_

    # Pair each class with its probability, sorted highest first
    prob_breakdown = sorted(
        zip(class_names, probabilities),
        key=lambda pair: pair[1],
        reverse=True
    )

    confidence = float(max(probabilities)) * 100

    return {
        "predicted_condition": predicted_class,
        "confidence_percent": round(confidence, 2),
        "probability_breakdown": [
            {"condition": name, "probability_percent": round(float(p) * 100, 2)}
            for name, p in prob_breakdown
        ]
    }
