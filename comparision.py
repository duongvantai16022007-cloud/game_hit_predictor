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
from sklearn.metrics import accuracy_score, classification_report

def load_and_process_data():
    data_file = 'dataset.csv'
    df = pd.read_csv(data_file)
    y = df['Hit_Target']
    slp = 30 
    df['Publisher'] = df['Publisher'].astype(str)
    top_pubs = df['Publisher'].value_counts().nlargest(slp).index
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')
    pub_counts = df['Publisher'].value_counts()
    df['Publisher_Experience'] = df['Publisher'].map(pub_counts).apply(np.log1p)
    num_cols = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience']
    cat_cols = ['Platform', 'Genre', 'Publisher_Group'] 
    existing_num = [c for c in num_cols if c in df.columns]
    existing_cat = [c for c in cat_cols if c in df.columns]
    X_num = df[existing_num].copy()
    X_cat = df[existing_cat].copy()
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

    print("🔄 Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_num)  
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_enc = encoder.fit_transform(X_cat)
    X_final = np.hstack((scaled, X_cat_enc))
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
def run_comparison():
    X_train, X_test, y_train, y_test = load_and_process_data()
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
        print(f"✅ {name:<25} | {acc*100:.2f}%     | {duration:.4f}s")
        print("-" * 40)
run_comparison()