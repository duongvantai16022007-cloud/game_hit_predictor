import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
def load_and_process_data():
    data = 'dataset.csv'
    slp = 10
    df = pd.read_csv(data)
    y = df['Hit_Target']
    top_pubs = df['Publisher'].value_counts().nlargest(slp).index
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')
    num = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience']
    cate = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']
    X_num = df[num]
    X_cat = df[cate]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_num)  
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_enc = encoder.fit_transform(X_cat)
    X_final = np.hstack((scaled, X_cat_enc))
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

def run_comparison():
    X_train, X_test, y_train, y_test = load_and_process_data()
    if X_train is None: return
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42)
    }

    results = []
    print(f"{'Model':<30} | {'acc':<15} | {'Time'}")
    print("-" * 65)
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        duration = end - start
        print(f"{name:<30} | {acc*100:.2f}%          | {duration:.4f}s")
        results.append({'Model': name, 'Accuracy': acc, 'Time': duration})
        report_str = classification_report(y_test, y_pred, target_names=['FLOP', 'HIT'], zero_division=0)
        print(report_str)
        print("-" * 65)

run_comparison()