import pandas as pd
import numpy as np
import time
import joblib
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = 'game_model_pipeline.pkl'
DATA_PATH = 'dataset.csv'

def load_and_engineer_features():
    df = pd.read_csv(DATA_PATH, encoding='utf-8', encoding_errors='replace')
    
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)
    for col in ['Critic_Score', 'User_Score', 'Global_Sales', 'Publisher_Experience']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    df['Name'] = df['Name'].astype(str)
    df['Name_Length'] = df['Name'].str.len()
    df['Word_Count'] = df['Name'].str.split().str.len()
    df['Is_Sequel'] = df['Name'].str.contains(r'\s\d+$|\s[IVX]+$|:', regex=True).astype(int)

    maker_map = {
        'Wii': 'Nintendo', 'NES': 'Nintendo', 'GB': 'Nintendo', 'DS': 'Nintendo',
        '3DS': 'Nintendo', 'Switch': 'Nintendo', 'WiiU': 'Nintendo', 'GBA': 'Nintendo',
        'SNES': 'Nintendo', 'N64': 'Nintendo', 'GC': 'Nintendo',
        'PS': 'Sony', 'PS2': 'Sony', 'PS3': 'Sony', 'PS4': 'Sony', 'PS5': 'Sony', 'PSP': 'Sony', 'PSV': 'Sony',
        'X360': 'Microsoft', 'XB': 'Microsoft', 'XOne': 'Microsoft', 'XSeries': 'Microsoft',
        'PC': 'PC'
    }
    df['Console_Maker'] = df['Platform'].map(maker_map).fillna('Other')
    year_counts = df['Year'].value_counts().sort_index()
    df['Competition_Index'] = df['Year'].map(year_counts.shift(1).fillna(0))
    df = df[df['Year'] >= 2000].copy()
    df['Hit_Target'] = (df['Global_Sales'] >= 0.2).astype(int)
    top_pubs = df['Publisher'].value_counts().nlargest(20).index
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')
    return df

def train_new_model():
    df = load_and_engineer_features()
    num_features = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience', 'Critic_Score', 'User_Score']
    cat_features = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']
    X = df[num_features + cat_features]
    y = df['Hit_Target']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    time_run = time.time() - start_time
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=['FLOP', 'HIT'], zero_division=0)
    return {
        'pipeline': pipeline,
        'acc': acc,
        'time_run': time_run,
        'report_str': report_str,
        'ui_metadata': {
            'platforms': sorted(df['Platform'].unique().tolist()),
            'publishers': sorted(df['Publisher'].unique().tolist()),
            'genres': sorted(df['Genre'].unique().tolist())
        }
    }

def get_model_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        artifacts = train_new_model()
        joblib.dump(artifacts, MODEL_PATH)
        return artifacts

def add_feedback_and_retrain(input_data, actual_label):
    new_row = pd.DataFrame([{
        'Name': str(input_data['Name']),
        'Platform': str(input_data['Platform']),
        'Year': int(input_data['Year']),
        'Genre': str(input_data['Genre']),
        'Publisher': str(input_data['Publisher']),
        'Global_Sales': 1.0 if actual_label == 1 else 0.01,
        'Critic_Score': 0,
        'User_Score': 0,
        'Publisher_Experience': float(input_data.get('Publisher_Experience', 0)) 
    }])

    if os.path.exists(DATA_PATH):
        df_old = pd.read_csv(DATA_PATH)
        df_new = pd.concat([df_old, new_row], ignore_index=True)
        df_new.to_csv(DATA_PATH, index=False)
    else:
        new_row.to_csv(DATA_PATH, index=False)
    artifacts = train_new_model()
    joblib.dump(artifacts, MODEL_PATH)
    return artifacts
