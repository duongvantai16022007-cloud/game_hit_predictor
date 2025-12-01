import streamlit as st
import pandas as pd
import numpy as np
import re
from model import model_rf 

st.set_page_config(page_title="Game Hit Predictor", layout="wide")
@st.cache_resource
def load_resources():
    return model_rf()
@st.cache_data
def load_raw_data():
    return pd.read_csv('dataset.csv')
def get_console_maker(platform):
    maker_map = {
        'Wii': 'Nintendo', 'NES': 'Nintendo', 'GB': 'Nintendo', 'DS': 'Nintendo', 
        '3DS': 'Nintendo', 'Switch': 'Nintendo',
        'PS': 'Sony', 'PS2': 'Sony', 'PS3': 'Sony', 'PS4': 'Sony', 'PS5': 'Sony', 'PSP': 'Sony',
        'X360': 'Microsoft', 'XB': 'Microsoft', 'XOne': 'Microsoft', 'PC': 'PC'
    }
    return maker_map.get(platform, 'Other')
def main():
    st.title("Predict Game Hit/Flop")
    st.markdown("This project uses the Random Forest algorithm to predict whether a game will be a hit or a flop.")
    model, scaler, encoder, num_cols, cat_cols, time_run, acc, report_str = load_resources()
    df_raw = load_raw_data()
    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.info(f"Training time: {time_run:.4f} giây")
    with col_metrics2:
        st.success(f"Accuracy: {acc*100:.2f}%")
    with st.expander("Classification Report"):
        st.code(report_str)
    st.divider() # chia làm 2 cột 
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Game", "Grand theft auto VI")
        year = st.number_input("Year", 2000, 2030, 2025)
        plat_opts = sorted(df_raw['Platform'].unique()) if df_raw is not None else ['PC']
        platform = st.selectbox("Platform", plat_opts)
        is_sequel_str = st.radio("Is this sequel", ("Yes", "No"), index=1, horizontal=True, help="Determine if this game has sequels or belongs to a famous series.")
        is_sequel = 1 if is_sequel_str == "Yes" else 0
    with c2:
        pub_opts = sorted(df_raw['Publisher'].unique()) if df_raw is not None else ['Nintendo']
        publisher = st.selectbox("Publisher", pub_opts)
        genre_opts = sorted(df_raw['Genre'].unique()) if df_raw is not None else ['Action']
        genre = st.selectbox("Genre", genre_opts)
    # logic khi nhấn 
    if st.button("Predict", type="primary", use_container_width=True):
        name_len = len(name)
        word_count = len(name.split())
        console_maker = get_console_maker(platform)
        if year in df_raw['Year'].values:
            comp_index = df_raw[df_raw['Year'] == year]['Competition_Index'].iloc[0]
        else:
            unique_years = df_raw['Year'].unique()
            top_5_years = sorted(unique_years, reverse=True)[:5] 
            recent_games = df_raw[df_raw['Year'].isin(top_5_years)]
            recent_comp = recent_games['Competition_Index'].mean()
            comp_index = recent_comp
        raw_count = df_raw['Publisher'].value_counts()[publisher]
        pub_exp = np.log1p(raw_count) 
        top_pubs = df_raw['Publisher'].value_counts().nlargest(20).index
        pub_group = publisher if publisher in top_pubs else 'Other'
        input_data = pd.DataFrame({
            'Year': [year], 'Name_Length': [name_len], 'Is_Sequel': [is_sequel],
            'Word_Count': [word_count], 'Competition_Index': [comp_index], 
            'Publisher_Experience': [pub_exp],
            'Platform': [platform], 'Genre': [genre], 
            'Console_Maker': [console_maker], 'Publisher_Group': [pub_group]
        })
        X_new_num = input_data[num_cols]
        X_new_cat = input_data[cat_cols]
        X_new_num_scaled = scaler.transform(X_new_num)
        X_new_cat_encoded = encoder.transform(X_new_cat)
        X_final = np.hstack((X_new_num_scaled, X_new_cat_encoded))
        pred = model.predict(X_final)[0]
        st.divider()
        if pred == 1:
            st.success(f"HIT")
        else:
            st.error(f"FLOP")

main()