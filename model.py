import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
data = 'dataset.csv'
slp = 30
def get_top_publishers(df, limit=30):
    return df['Publisher'].value_counts().nlargest(limit).index.tolist()
def model_rf():
    df = pd.read_csv(data)
    y = df['Hit_Target']
    top_pubs = get_top_publishers(df, limit=slp)
    df['Publisher'] = df['Publisher'].astype(str)
    df['Publisher_Group'] = df['Publisher'].apply(lambda x: x if x in top_pubs else 'Other')
    pub_counts = df['Publisher'].value_counts()
    df['Publisher_Experience'] = df['Publisher'].map(pub_counts).apply(np.log1p)
    num = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience']
    cate = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']
    for col in num:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in cate:
        if col in df.columns:
             df[col] = df[col].fillna('Unknown').astype(str)
        else:
             df[col] = 'Unknown'
    X_num = df[num]
    X_cat = df[cate]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_num)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_enc = encoder.fit_transform(X_cat)
    X_final = np.hstack((scaled, X_cat_enc))
    X_final = np.nan_to_num(X_final)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    end_time = time.time()
    time_run = end_time - start_time
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    report_str = classification_report(y_test, y_pred, target_names=['FLOP', 'HIT'], zero_division=0)
    return model, scaler, encoder, num, cate, time_run, acc, report_str, X_train, y_train