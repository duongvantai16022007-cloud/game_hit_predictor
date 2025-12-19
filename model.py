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
ENRICHED_DATA_PATH = 'dataset_final_processed.csv'


def load_and_engineer_features():
    if os.path.exists(ENRICHED_DATA_PATH):
        df = pd.read_csv(ENRICHED_DATA_PATH, encoding='utf-8', encoding_errors='replace')
        cols = ['Critic_Score', 'User_Score', 'Total_Rating_Count']
        for c in cols: df[c] = df.get(c, 0).fillna(0)
        has_enrichment = True
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding='utf-8', encoding_errors='replace')
        df['Critic_Score'] = 0
        df['User_Score'] = 0
        df['Total_Rating_Count'] = 0
        has_enrichment = False
    else:
        raise FileNotFoundError("No dataset found!")

    df['Name'] = df['Name'].astype(str)
    df['Name_Length'] = df['Name'].apply(len)
    df['Word_Count'] = df['Name'].apply(lambda x: len(x.split()))

    def is_sequel(name):
        name = name.lower()
        if re.search(r'\s\d+$', name): return 1
        if re.search(r'\s[IVX]+$', name): return 1
        if ':' in name: return 1
        return 0

    df['Is_Sequel'] = df['Name'].apply(is_sequel)

    def get_console_maker(platform):
        maker_map = {
            'Wii': 'Nintendo', 'NES': 'Nintendo', 'GB': 'Nintendo', 'DS': 'Nintendo',
            '3DS': 'Nintendo', 'Switch': 'Nintendo', 'WiiU': 'Nintendo', 'GBA': 'Nintendo',
            'SNES': 'Nintendo', 'N64': 'Nintendo', 'GC': 'Nintendo',
            'PS': 'Sony', 'PS2': 'Sony', 'PS3': 'Sony', 'PS4': 'Sony', 'PS5': 'Sony', 'PSP': 'Sony', 'PSV': 'Sony',
            'X360': 'Microsoft', 'XB': 'Microsoft', 'XOne': 'Microsoft', 'XSeries': 'Microsoft',
            'PC': 'PC'
        }
        return maker_map.get(platform, 'Other')

    df['Console_Maker'] = df['Platform'].apply(get_console_maker)

    df = df.sort_values(by=['Year', 'Publisher'])
    df['Publisher_Experience'] = df.groupby('Publisher').cumcount()
    df['Publisher_Experience'] = np.log1p(df['Publisher_Experience'])

    year_counts = df['Year'].value_counts().sort_index()
    lagged_counts = year_counts.shift(1).fillna(0)
    df['Competition_Index'] = df['Year'].map(lagged_counts)

    df = df[df['Year'] >= 2000].copy()

    if has_enrichment:
        df['Hit_Target'] = ((df['Global_Sales'] >= 0.2) | (df['Total_Rating_Count'] > 500)).astype(int)
    else:
        df['Hit_Target'] = (df['Global_Sales'] >= 0.2).astype(int)

    top_pubs = df['Publisher'].value_counts().nlargest(20).index
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')

    return df


def train_new_model():
    df = load_and_engineer_features()

    num_features = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience',
                    'Critic_Score', 'User_Score']
    cat_features = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']

    X = df[num_features + cat_features]
    y = df['Hit_Target']

    preprocessor = ColumnTransformer(
        transformers=[
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

    ui_metadata = {
        'platforms': sorted(df['Platform'].unique().tolist()),
        'publishers': sorted(df['Publisher'].unique().tolist()),
        'genres': sorted(df['Genre'].unique().tolist()),
    }

    artifacts = {
        'pipeline': pipeline,
        'acc': acc,
        'time_run': time_run,
        'report_str': report_str,
        'ui_metadata': ui_metadata
    }

    return artifacts


def get_model_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        artifacts = train_new_model()
        joblib.dump(artifacts, MODEL_PATH)
        return artifacts


def add_feedback_and_retrain(input_data, actual_label):
    target_file = ENRICHED_DATA_PATH if os.path.exists(ENRICHED_DATA_PATH) else DATA_PATH

    df = pd.read_csv(target_file)

    dummy_sales = 1.0 if actual_label == 1 else 0.01

    new_row = {
        'Name': input_data['Name'],
        'Platform': input_data['Platform'],
        'Year': input_data['Year'],
        'Genre': input_data['Genre'],
        'Publisher': input_data['Publisher'],
        'Global_Sales': dummy_sales,
        'Critic_Score': 0,
        'User_Score': 0,
        'Total_Rating_Count': 0
    }

    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(target_file, index=False)

    artifacts = train_new_model()
    joblib.dump(artifacts, MODEL_PATH)

    return artifacts


if __name__ == "__main__":
    results = get_model_pipeline()
    print(f"Accuracy: {results['acc']:.2%}")