import streamlit as st
import pandas as pd
import numpy as np
import re
import shap
import matplotlib.pyplot as plt
from model import model_rf 

st.set_page_config(page_title="Game Hit Predictor AI", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button {
        background: linear-gradient(to right, #ff4b4b, #ff0000);
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
</style>
""", unsafe_allow_html=True)

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
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image("https://cdn-icons-png.flaticon.com/512/3408/3408506.png", width=100)
    with c2:
        st.title("AI Game Hit Predictor")
        st.caption("Hệ thống hỗ trợ ra quyết định đầu tư Game bằng Machine Learning")

    model, scaler, encoder, num_cols, cat_cols, time_run, acc, report_str, _, _ = load_resources()
    df_raw = load_raw_data()

    col_input, col_output = st.columns([1, 1.5], gap="large")
    with col_input:
        st.subheader("Thông số dự án cơ bản")
        with st.container():
            name = st.text_input("Tên dự án Game", "Super Mario 2026")
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                year = st.number_input("Năm phát hành", 2000, 2030, 2025)
                plat_opts = sorted(df_raw['Platform'].unique()) if df_raw is not None else ['PC']
                platform = st.selectbox("Hệ máy (Platform)", plat_opts)
                is_sequel_str = st.radio("Là phần tiếp theo?", ("Có", "Không"), index=1, horizontal=True)
                is_sequel = 1 if is_sequel_str == "Có" else 0
            with c_in2:
                pub_opts = sorted(df_raw['Publisher'].unique()) if df_raw is not None else ['Nintendo']
                publisher = st.selectbox("Nhà phát hành", pub_opts)
                genre_opts = sorted(df_raw['Genre'].unique()) if df_raw is not None else ['Action']
                genre = st.selectbox("Thể loại", genre_opts)
            st.write("")
            predict_btn = st.button("PHÂN TÍCH TIỀM NĂNG", type="primary", use_container_width=True)
    if predict_btn:
        name_len = len(name)
        word_count = len(name.split())
        console_maker = get_console_maker(platform)
        if year in df_raw['Year'].values:
            comp_index = df_raw[df_raw['Year'] == year]['Competition_Index'].iloc[0]
        else:
            unique_years = sorted(df_raw['Year'].unique()) 
            last_5_years = unique_years[-5:]               
            recent_data = df_raw[df_raw['Year'].isin(last_5_years)]
            comp_index = recent_data['Competition_Index'].mean()
            comp_source = "Dự báo (TB 5 năm)"
        
        if publisher in df_raw['Publisher'].values:
            raw_count = df_raw['Publisher'].value_counts()[publisher]
            pub_exp = np.log1p(raw_count)
        else:
            pub_exp = 0            
        pub_group = publisher if (df_raw is not None and publisher in df_raw['Publisher'].value_counts().nlargest(20).index) else 'Other'
        input_data = pd.DataFrame({
            'Year': [year], 'Name_Length': [name_len], 'Is_Sequel': [is_sequel],
            'Word_Count': [word_count], 'Competition_Index': [comp_index], 
            'Publisher_Experience': [pub_exp],
            'Platform': [platform], 'Genre': [genre], 
            'Console_Maker': [console_maker], 'Publisher_Group': [pub_group]
        })
        X_new_num = input_data[num_cols]
        X_new_cat = input_data[cat_cols]
        X_final = np.hstack((scaler.transform(X_new_num), encoder.transform(X_new_cat)))
        pred = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0]
        with col_output:
            st.subheader("Kết quả Phân tích")
            tab_result, tab_explain, tab_stats = st.tabs(["🎯 Kết quả", "🔍 Giải thích", "📈 Thống kê Model"])
            with tab_result:
                m1, m2, m3 = st.columns(3)
                m1.metric("Dự đoán", "HIT" if pred == 1 else "FLOP", delta="Thành công" if pred==1 else "-Rủi ro")
                m2.metric("Độ tin cậy", f"{proba[pred]*100:.1f}%")
                st.write("Xác suất thành công:")
                st.progress(int(proba[1]*100))
                if pred == 1:
                    st.success(f"Chúc mừng! **{name}** có tiềm năng trở thành bom tấn toàn cầu.")
                else:
                    st.warning(f"Cảnh báo! **{name}** có rủi ro thất bại cao. Cần cân nhắc lại chiến lược.")
                m3.metric("⚔️ Số Game Cạnh Tranh", f"{int(comp_index)}", delta=comp_source, delta_color="off", help=f"Ước tính có khoảng {int(comp_index)} game cùng phát hành trong năm {year}.")
            with tab_explain:
                st.write("Các yếu tố ảnh hưởng lớn nhất đến kết quả này:")
                with st.spinner("..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_final, check_additivity=False)                   
                    if isinstance(shap_values, list):
                        vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    else:
                        vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#ff4b4b' if x > 0 else '#1f77b4' for x in vals]
                    indices = np.argsort(np.abs(vals))[-8:]
                    ax.barh(range(len(indices)), vals[indices], color=[colors[i] for i in indices])
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
                    ax.set_xlabel("Mức độ tác động (+HIT / -FLOP)")
                    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
            with tab_stats:
                st.info(f"⏱ Thời gian huấn luyện Model: **{time_run:.4f} giây**")
                st.info(f"Model Accuracy: **{acc*100:.2f}%**")
                st.code(report_str, language='text')

main()