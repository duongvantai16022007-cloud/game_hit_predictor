import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import model_rf
def visualize_feature_importance(model, encoder, num_cols, cat_cols):
    feature_names = list(num_cols)
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    all_features = feature_names + cat_feature_names
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    })
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
    plt.title('TOP 20 MOST IMPORTANT FEATURES', fontsize=15, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

results = model_rf()
model = results[0]
encoder = results[2]
num_cols = results[3]
cat_cols = results[4]

visualize_feature_importance(model, encoder, num_cols, cat_cols)
