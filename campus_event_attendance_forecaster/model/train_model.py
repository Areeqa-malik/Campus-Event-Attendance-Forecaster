"""
train_model.py
---------------
Trains a scikit-learn RandomForestRegressor to predict event attendance from
event features (type, day, time slot, weather, marketing effort, social media
reach, venue capacity, holiday flag, competing event flag, past average
attendance).

This is the "AI" component of the project:
    1. Categorical features (event_type, day_of_week, time_slot, weather,
       marketing_effort) are one-hot encoded.
    2. Numerical features are used as-is.
    3. A Random Forest (an ensemble of decision trees) learns the non-linear
       relationships between these features and real attendance figures.
    4. The trained model + the exact column layout it expects are saved to
       disk with joblib, so the GUI app can load them instantly without
       retraining every time.

Run directly:
    python train_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "campus_events.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "attendance_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

CATEGORICAL_COLS = ["event_type", "day_of_week", "time_slot", "weather", "marketing_effort"]
NUMERICAL_COLS = [
    "social_media_reach", "venue_capacity", "is_holiday",
    "competing_event", "past_avg_attendance",
]
TARGET_COL = "attendance"


def ensure_data_exists():
    if not os.path.exists(DATA_PATH):
        print("No dataset found. Generating synthetic historical data first...")
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        import generate_data
        generate_data.main()


def build_features(df: pd.DataFrame, training_columns=None):
    """One-hot encode categorical columns and align numeric columns.
    If training_columns is provided, the output is re-indexed to match it
    exactly (this is essential so the GUI's single-row prediction has the
    same columns, in the same order, as what the model was trained on)."""
    encoded = pd.get_dummies(df[CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
    features = pd.concat([encoded, df[NUMERICAL_COLS]], axis=1)

    if training_columns is not None:
        features = features.reindex(columns=training_columns, fill_value=0)

    return features


def main():
    ensure_data_exists()
    df = pd.read_csv(DATA_PATH)

    X = build_features(df)
    y = df[TARGET_COL]

    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=" * 50)
    print("Model Evaluation on Held-Out Test Data")
    print("=" * 50)
    print(f"Mean Absolute Error : {mae:.2f} attendees")
    print(f"R^2 Score           : {r2:.3f}")
    print("=" * 50)

    # Feature importance (nice to show on LinkedIn / in your report!)
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=False)
    print("\nTop features driving attendance predictions:")
    print(importances.head(8).round(3))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, COLUMNS_PATH)
    print(f"\nModel saved to      : {MODEL_PATH}")
    print(f"Feature list saved to: {COLUMNS_PATH}")


if __name__ == "__main__":
    main()
