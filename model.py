import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Define where to save the specific model artifacts
MODEL_PATH = 'game_model_artifacts.joblib'


def train_new_model():
    """
    Performs the heavy lifting: Loading data, training, and calculating metrics.
    Returns a dictionary of artifacts.
    """
    start_time = time.time()

    # 1. Load Data
    df = pd.read_csv('dataset.csv')

    # 2. Preprocessing Logic (Reconstructed from your comparison.py)
    slp = 10
    top_pubs = df['Publisher'].value_counts().nlargest(slp).index
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')

    num_cols = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience']
    cat_cols = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']

    X_num = df[num_cols]
    X_cat = df[cat_cols]
    y = df['Hit_Target']

    # 3. Transformers
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_num)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_enc = encoder.fit_transform(X_cat)

    X_final = np.hstack((scaled, X_cat_enc))

    # 4. Split and SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 5. Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # 6. Evaluate
    end_time = time.time()
    time_run = end_time - start_time

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=['FLOP', 'HIT'], zero_division=0)

    # 7. Pack everything into a dictionary
    artifacts = {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'time_run': time_run,
        'acc': acc,
        'report_str': report_str,
        # We don't necessarily need to save the training data for inference,
        # but your main.py unpacks it, so we can store it or dummies.
        # Storing None to save disk space as main.py ignores them anyway.
        'X_train_resampled': None,
        'y_train_resampled': None
    }

    return artifacts


def model_rf():
    """
    The main interface used by Streamlit.
    Checks for cached model first. If missing, trains a new one.
    """
    # CHECK: Does the file exist?
    if os.path.exists(MODEL_PATH):
        # LOAD: Fast path
        print(f"Loading model from {MODEL_PATH}...")  # Optional: for your console logs
        artifacts = joblib.load(MODEL_PATH)
    else:
        # TRAIN: Slow path
        print("Cached model not found. Training new model...")
        artifacts = train_new_model()
        # SAVE: Create the cache for next time
        joblib.dump(artifacts, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # UNPACK: Return exactly 10 values as expected by main.py
    return (
        artifacts['model'],
        artifacts['scaler'],
        artifacts['encoder'],
        artifacts['num_cols'],
        artifacts['cat_cols'],
        artifacts['time_run'],
        artifacts['acc'],
        artifacts['report_str'],
        artifacts['X_train_resampled'],
        artifacts['y_train_resampled']
    )