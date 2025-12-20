import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import time
import os
from model import get_model_pipeline, add_feedback_and_retrain

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
    return get_model_pipeline()

@st.cache_data
def load_lookup_data():
    if os.path.exists('dataset.csv'):
        return pd.read_csv('dataset.csv', usecols=['Year', 'Publisher', 'Competition_Index', 'Publisher_Experience'])
    return pd.DataFrame(columns=['Year', 'Publisher', 'Competition_Index', 'Publisher_Experience'])

def get_console_maker(platform):
    maker_map = {
        'Wii': 'Nintendo', 'NES': 'Nintendo', 'GB': 'Nintendo', 'DS': 'Nintendo',
        '3DS': 'Nintendo', 'Switch': 'Nintendo', 'WiiU': 'Nintendo', 'GBA': 'Nintendo',
        'PS': 'Sony', 'PS2': 'Sony', 'PS3': 'Sony', 'PS4': 'Sony', 'PS5': 'Sony', 'PSP': 'Sony', 'PSV': 'Sony',
        'X360': 'Microsoft', 'XB': 'Microsoft', 'XOne': 'Microsoft', 'XSeries': 'Microsoft',
        'PC': 'PC'
    }
    return maker_map.get(platform, 'Other')

def get_shap_contribution(explainer, X_transformed):
    shap_values = explainer.shap_values(X_transformed, check_additivity=False)
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            vals = shap_values[1]
        else:
            vals = shap_values[0]
    else:
        vals = shap_values
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    if vals.ndim == 2:
        vals = vals[0]
    return vals

def predict_and_update_state(pipeline, input_df, raw_inputs, current_name, comp_index, comp_source):
    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]

    st.session_state['show_results'] = True
    st.session_state['pred_result'] = pred
    st.session_state['pred_proba'] = proba
    st.session_state['input_df'] = input_df
    st.session_state['raw_inputs'] = raw_inputs
    st.session_state['current_name'] = current_name
    st.session_state['current_comp_index'] = comp_index
    st.session_state['current_comp_source'] = comp_source

def main():
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image("https://cdn-icons-png.flaticon.com/512/3408/3408506.png", width=100)
    with c2:
        st.title("AI Game Hit Predictor")
        st.caption("Hệ thống hỗ trợ ra quyết định đầu tư Game")

    data = load_resources()
    pipeline = data['pipeline']
    ui_data = data['ui_metadata']

    col_input, col_output = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("Thông số dự án")
        with st.container():
            name = st.text_input("Tên dự án Game", "Super Mario 2026")

            c_in1, c_in2 = st.columns(2)
            with c_in1:
                year = st.number_input("Năm phát hành", 2000, 2030, 2025)
                platform = st.selectbox("Hệ máy (Platform)", ui_data['platforms'])
                is_sequel_str = st.radio("Là phần tiếp theo?", ("Có", "Không"), index=1, horizontal=True)
                is_sequel = 1 if is_sequel_str == "Có" else 0

            with c_in2:
                publisher = st.selectbox("Nhà phát hành", ui_data['publishers'])
                genre = st.selectbox("Thể loại", ui_data['genres'])

            with st.expander("⭐ Giả lập Đánh giá (Scenario Testing)"):
                st.info("Kéo thanh trượt để giả lập điểm số review (0 = Chưa có đánh giá)")
                critic_score = st.slider("Critic Score (Metacritic)", 0, 100, 0)
                user_score = st.slider("User Score (IGDB/Steam)", 0, 100, 0)

            st.write("")
            predict_btn = st.button("PHÂN TÍCH TIỀM NĂNG", type="primary", use_container_width=True)

    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = False

    if predict_btn:
        df_lookup = load_lookup_data()
        
        if year in df_lookup['Year'].values:
            comp_index = df_lookup[df_lookup['Year'] == year]['Competition_Index'].mean()
            comp_source = "Dữ liệu thực tế"
        else:
            last_3_years = sorted(df_lookup['Year'].unique())[-3:]
            comp_index = df_lookup[df_lookup['Year'].isin(last_3_years)]['Competition_Index'].mean()
            comp_source = "Dự báo xu hướng"
        
        pub_exp = 0.0
        if publisher in df_lookup['Publisher'].values:
            pub_exp = df_lookup[df_lookup['Publisher'] == publisher]['Publisher_Experience'].max()

        name_len = len(name)
        word_count = len(name.split())
        console_maker = get_console_maker(platform)

        raw_inputs = {
            'Name': name, 'Year': year, 'Platform': platform,
            'Publisher': publisher, 'Genre': genre,
            'Publisher_Experience': pub_exp
        }

        input_df = pd.DataFrame({
            'Year': [year], 'Name_Length': [name_len], 'Is_Sequel': [is_sequel],
            'Word_Count': [word_count], 'Competition_Index': [comp_index],
            'Publisher_Experience': [pub_exp], 'Platform': [platform],
            'Genre': [genre], 'Console_Maker': [console_maker],
            'Publisher_Group': [publisher], 'Critic_Score': [critic_score], 'User_Score': [user_score]
        })

        predict_and_update_state(pipeline, input_df, raw_inputs, name, comp_index, comp_source)

    if st.session_state['show_results']:
        pred = st.session_state['pred_result']
        proba = st.session_state['pred_proba']
        input_df = st.session_state['input_df']
        raw_inputs = st.session_state['raw_inputs']
        current_name = st.session_state['current_name']
        comp_index = st.session_state['current_comp_index']
        comp_source = st.session_state['current_comp_source']

        with col_output:
            st.subheader("Kết quả Phân tích")

            f1, f2 = st.columns(2)

            def handle_feedback(actual_label):
                with st.spinner("Đang học từ phản hồi và huấn luyện lại..."):
                    add_feedback_and_retrain(raw_inputs, actual_label)
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    new_data = load_resources()
                    new_pipeline = new_data['pipeline']
                    
                    predict_and_update_state(new_pipeline, input_df, raw_inputs, current_name, comp_index, comp_source)
                    st.success("Đã cập nhật Model! Kết quả bên dưới là dự đoán mới nhất.")
                    time.sleep(1.5)
                    st.rerun()

            with f1:
                if st.button("👎 Sai! Đây là FLOP"):
                    handle_feedback(0)
            with f2:
                if st.button("👎 Sai! Đây là HIT"):
                    handle_feedback(1)

            tab_result, tab_explain, tab_stats = st.tabs(["🎯 Kết quả", "🔍 Giải thích", "📈 Thống kê"])

            with tab_result:
                m1, m2, m3, m4 = st.columns(4)
                
                m1.metric("Dự đoán", "HIT" if pred == 1 else "FLOP", delta="Thành công" if pred == 1 else "-Rủi ro")
                m2.metric("Độ tin cậy", f"{proba[pred] * 100:.1f}%")
                m3.metric("Cạnh tranh", f"{int(comp_index)}", delta=comp_source, delta_color="off")
                m4.metric("Độ chính xác của model", f"{data['acc'] * 100:.1f}%")

                st.progress(int(proba[1] * 100))
                
                if pred == 1:
                    st.success(f"**{current_name}** có tiềm năng thành công.")
                else:
                    st.error(f"**{current_name}** có rủi ro cao.")

            with tab_explain:
                preprocessor = pipeline.named_steps['preprocessor']
                classifier = pipeline.named_steps['classifier']
                X_transformed = preprocessor.transform(input_df)
                explainer = shap.TreeExplainer(classifier)
                vals = get_shap_contribution(explainer, X_transformed)

                cat_cols = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']
                num_features_order = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index',
                                      'Publisher_Experience', 'Critic_Score', 'User_Score']
                ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
                feature_names = num_features_order + list(ohe_feature_names)

                if 'Critic_Score' in feature_names and 'User_Score' in feature_names:
                    critic_idx = feature_names.index('Critic_Score')
                    user_idx = feature_names.index('User_Score')
                    st.write("### 📊 Tác động điểm số:")
                    k1, k2 = st.columns(2)
                    k1.metric("Critic Impact", f"{vals[critic_idx]:+.3f}")
                    k2.metric("User Impact", f"{vals[user_idx]:+.3f}")

                fig, ax = plt.subplots(figsize=(8, 5))
                indices = np.argsort(np.abs(vals))[-8:]
                colors = ['#ff4b4b' if x > 0 else '#1f77b4' for x in vals[indices]]
                bars = ax.barh(range(len(indices)), vals[indices], color=colors)
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    label_x_pos = width if width > 0 else width - 0.02
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{vals[indices][i]:.2f}', va='center', fontsize=9)

                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
                ax.set_xlabel("Mức độ tác động (+HIT / -FLOP)")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

            with tab_stats:
                st.info(f"Training Time: {data['time_run']:.4f}s")
                st.code(data['report_str'])

main()