import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_and_process_data():
    df = pd.read_csv('dataset.csv', encoding='utf-8', encoding_errors='replace')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)
    for col in ['Critic_Score', 'User_Score', 'Global_Sales']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    df['Hit_Target'] = (df['Global_Sales'] >= 0.2).astype(int)
    y = df['Hit_Target']
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
    df = df.sort_values(by=['Year', 'Publisher'])
    df['Publisher_Experience'] = np.log1p(df.groupby('Publisher').cumcount())
    year_counts = df['Year'].value_counts().sort_index()
    df['Competition_Index'] = df['Year'].map(year_counts.shift(1).fillna(0))
    df = df[df['Year'] >= 2000].copy()
    y = df['Hit_Target'] 
    top_pubs = df['Publisher'].value_counts().nlargest(20).index
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')
    num_cols = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience', 'Critic_Score', 'User_Score']
    cat_cols = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']
    X_num = df[num_features] if 'num_features' in locals() else df[num_cols]
    X_cat = df[cat_cols]
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_enc = encoder.fit_transform(X_cat)
    X_final = np.hstack((X_num_scaled, X_cat_enc))
    return train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
def run_comparison():
    data = load_and_process_data()
    X_train, X_test, y_train, y_test = data
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced', cache_size=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    }
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Time (s)':<10}")
    print("-" * 80)
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        duration = end - start  
        print(f"{name:<25} | {acc*100:.2f}%     | {duration:.4f}s")
        print("-" * 80)

run_comparison()
