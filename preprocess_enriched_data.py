import pandas as pd
import numpy as np
import os

INPUT_FILE = 'dataset_enriched_igdb.csv'
CHECKPOINT_FILE = 'igdb_enriched_temp.csv'
ORIGINAL_DATA = 'dataset.csv'
OUTPUT_FILE = 'dataset_final_processed.csv'


def clean_and_finalize_data():
    if os.path.exists(INPUT_FILE):
        print(f"🚀 Loading complete enriched file: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8', encoding_errors='replace')
    elif os.path.exists(CHECKPOINT_FILE) and os.path.exists(ORIGINAL_DATA):
        print(f"⚠️ {INPUT_FILE} not found. Rescuing data from checkpoint: {CHECKPOINT_FILE}...")

        df_orig = pd.read_csv(ORIGINAL_DATA, encoding='utf-8', encoding_errors='replace')
        df_temp = pd.read_csv(CHECKPOINT_FILE, encoding='utf-8', encoding_errors='replace')

        df = pd.merge(df_orig, df_temp, on='Name', how='left')

        new_cols = ['Critic_Score', 'User_Score', 'Total_Rating_Count', 'IGDB_Genres']
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan

        print(f"   Merged {len(df_temp)} enriched rows into dataset.")
    else:
        print(f"❌ Error: Could not find {INPUT_FILE} or {CHECKPOINT_FILE}")
        return

    orig_hits = len(df[df['Global_Sales'] >= 0.2])
    orig_flops = len(df[df['Global_Sales'] < 0.2])
    initial_rows = len(df)

    print("🎨 Merging Genres...")

    df['Genre'] = df['Genre'].fillna('Unknown')
    if 'IGDB_Genres' in df.columns:
        df['IGDB_Genres'] = df['IGDB_Genres'].fillna('')
    else:
        df['IGDB_Genres'] = ''

    def merge_genre_logic(row):
        vg_genre = str(row['Genre'])
        igdb_raw = str(row['IGDB_Genres'])

        if igdb_raw and igdb_raw != 'nan':
            igdb_list = igdb_raw.split('|')
        else:
            igdb_list = []

        if vg_genre in ['Misc', 'Unknown', 'nan', '']:
            if len(igdb_list) > 0:
                return igdb_list[0]
            return 'Misc'

        return vg_genre

    df['Genre'] = df.apply(merge_genre_logic, axis=1)
    print(f"   - Genres consolidated. 'Misc' categories reduced using IGDB data.")

    print("📊 Handling Missing Scores...")

    score_cols = ['Critic_Score', 'User_Score', 'Total_Rating_Count']
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan

    df['Total_Rating_Count'] = df['Total_Rating_Count'].fillna(0)

    target_score_cols = ['Critic_Score', 'User_Score']

    for col in target_score_cols:
        df[f'Has_{col}'] = df[col].notna().astype(int) & (df[col] > 0).astype(int)

        df[col] = df[col].replace(0, np.nan)

        medians = df.groupby('Genre')[col].transform('median')
        df[col] = df[col].fillna(medians)

        global_median = df[col].median()
        if pd.isna(global_median): global_median = 0

        df[col] = df[col].fillna(global_median)

        print(f"   - Imputed {col} using Genre Medians (Global fallback: {global_median:.1f})")

    df['Has_Critic_Score'] = df['Has_Critic_Score'].astype(int)
    df['Has_User_Score'] = df['Has_User_Score'].astype(int)

    df = df.dropna(subset=['Name', 'Year'])

    df.to_csv(OUTPUT_FILE, index=False)

    final_hits = len(df[df['Global_Sales'] >= 0.2])
    final_flops = len(df[df['Global_Sales'] < 0.2])

    print("-" * 30)
    print(f"✅ SUCCESS! Final dataset saved to: {OUTPUT_FILE}")
    print(f"\n📈 DATASET STATISTICS:")
    print(f"   Original Rows: {initial_rows} | Hits: {orig_hits} | Flops: {orig_flops}")
    print(f"   Final Rows:    {len(df)} | Hits: {final_hits} | Flops: {final_flops}")

    if 'Has_Critic_Score' in df.columns:
        real_scores = df['Has_Critic_Score'].sum()
        print(f"   Games with REAL Critic Scores: {real_scores} ({(real_scores / len(df)) * 100:.1f}%)")


if __name__ == "__main__":
    clean_and_finalize_data()