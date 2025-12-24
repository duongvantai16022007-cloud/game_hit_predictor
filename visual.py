import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import get_model_pipeline

def visualize_feature_importance():
    artifacts = get_model_pipeline()
    pipeline = artifacts['pipeline']
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    num_cols = ['Year', 'Name_Length', 'Is_Sequel', 'Word_Count', 'Competition_Index', 'Publisher_Experience', 'Critic_Score', 'User_Score']
    cat_cols = ['Platform', 'Genre', 'Console_Maker', 'Publisher_Group']
    ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_features = num_cols + list(ohe_features)
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=fi_df, 
        palette='viridis', 
        hue='Feature', 
        legend=False
    )
    plt.title('TOP 20 MOST IMPORTANT FEATURES (HIT/FLOP)', fontsize=15, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

visualize_feature_importance()
