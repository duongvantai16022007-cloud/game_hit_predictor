import requests
import pandas as pd
import time
import os
import re
from datetime import datetime
from fuzzywuzzy import fuzz  # Use fuzzywuzzy instead of difflib

# --- CONFIGURATION ---
# 🔴 PASTE YOUR KEYS HERE
CLIENT_ID = 'skg7afvjlq76k7z1d6l436xz4c1r9y'
CLIENT_SECRET = 'qf1l2r8i4t7uqh9e2hyqrgllc8oopw'

# Files
INPUT_CSV = 'dataset.csv'
OUTPUT_CSV = 'dataset_enriched_igdb.csv'
TEMP_CSV = 'igdb_enriched_temp.csv'

# Thresholds
YEAR_TOLERANCE = 1  # Matches if IGDB year is within +/- 1 year of CSV year
NAME_SIMILARITY_THRESHOLD = 60  # Fuzzywuzzy uses 0-100 scale, so 0.6 becomes 60


def get_access_token(client_id, client_secret):
    url = 'https://id.twitch.tv/oauth2/token'
    params = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }
    try:
        response = requests.post(url, params=params)
        response.raise_for_status()
        return response.json()['access_token']
    except Exception as e:
        print(f"🛑 AUTH ERROR: {e}")
        exit()


def clean_name_for_search(name):
    """Simplifies name for better search hits (removes special chars)."""
    name = re.sub(r'\(.*?\)', '', str(name))  # Remove (Year) or (Region)
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'[^a-zA-Z0-9\s:]', '', name)  # Keep colons for subtitles
    return name.strip()


def get_year_from_timestamp(ts):
    if not ts: return 0
    return datetime.fromtimestamp(ts).year


def find_best_match(target_name, target_year, candidates):
    """
    Logic:
    1. Filter candidates that are within Year Tolerance.
    2. From those, pick the one with highest Name Similarity using fuzzywuzzy.
    3. If no year match, fallback to Name Similarity but warn.
    """
    best_match = None
    best_score = 0

    # Clean target
    clean_target = clean_name_for_search(target_name).lower()

    # 1. Filter by Year (if target_year exists)
    year_matches = []
    if target_year > 0:
        for item in candidates:
            # IGDB returns first_release_date as unix timestamp
            igdb_year = get_year_from_timestamp(item.get('first_release_date'))
            if abs(igdb_year - target_year) <= YEAR_TOLERANCE:
                year_matches.append(item)

    # Use full list if no year matches found (Fallback)
    search_pool = year_matches if year_matches else candidates

    # 2. String Matching with fuzzywuzzy
    for item in search_pool:
        clean_candidate = clean_name_for_search(item['name']).lower()
        # token_sort_ratio handles "God of War: Ragnarok" vs "Ragnarok: God of War" better
        score = fuzz.token_sort_ratio(clean_target, clean_candidate)

        if score > best_score:
            best_score = score
            best_match = item

    return best_match, best_score, (len(year_matches) > 0)


def run_enrichment():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_CSV)
    # Ensure Year is int
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

    # Get unique games (Name + Year pair)
    unique_games = df[['Name', 'Year']].drop_duplicates(subset=['Name']).to_dict('records')
    total_games = len(unique_games)

    print(f"🚀 Starting Smart Enrichment for {total_games} games...")
    print(f"   Logic: FuzzyWuzzy Match Name + Year (+/- {YEAR_TOLERANCE} yr)")

    # 2. Check for Resume
    processed_games = {}
    if os.path.exists(TEMP_CSV):
        print("📂 Resume: Loading temp file...")
        temp_df = pd.read_csv(TEMP_CSV)
        for _, row in temp_df.iterrows():
            processed_games[row['Name']] = row.to_dict()
        print(f"   Skipping {len(processed_games)} already processed.")

    # 3. Setup API
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    headers = {
        'Client-ID': CLIENT_ID,
        'Authorization': f'Bearer {token}',
        'Content-Type': 'text/plain',
        'Accept': 'application/json'
    }
    url = "https://api.igdb.com/v4/games"

    new_results = []

    for i, game in enumerate(unique_games):
        name = game['Name']
        year = game['Year']

        if name in processed_games:
            continue

        clean_name = clean_name_for_search(name)

        # Search query: Get Top 5 results to filter locally
        # We fetch fields needed for matching (name, date) and data (ratings)
        body = f'search "{clean_name}"; fields name, first_release_date, total_rating, total_rating_count, aggregated_rating, aggregated_rating_count, genres.name; limit 10;'

        try:
            response = requests.post(url, headers=headers, data=body.encode('utf-8'))

            result = {
                'Name': name,
                'Critic_Score': 0,
                'User_Score': 0,
                'Total_Rating_Count': 0,
                'IGDB_Genres': '',
                'Match_Type': 'None',  # Debug info
                'IGDB_ID': 0
            }

            if response.status_code == 200:
                candidates = response.json()

                if candidates:
                    match, score, year_aligned = find_best_match(name, year, candidates)

                    if match and score >= NAME_SIMILARITY_THRESHOLD:
                        # Extract Data
                        result['Critic_Score'] = match.get('aggregated_rating', 0)
                        result['User_Score'] = match.get('total_rating', 0)
                        result['Total_Rating_Count'] = match.get('total_rating_count', 0)
                        result['IGDB_ID'] = match.get('id')

                        if 'genres' in match:
                            result['IGDB_Genres'] = "|".join([g['name'] for g in match['genres']])

                        # Debug Status
                        status_icon = "✅" if year_aligned else "⚠️"
                        result['Match_Type'] = 'Year_Exact' if year_aligned else 'Name_Only'

                        if i % 10 == 0:
                            print(f"{status_icon} [{i}/{total_games}] {name} -> {match['name']} (Score: {score})")
                    else:
                        if i % 50 == 0: print(f"❌ [{i}/{total_games}] No good match for: {name}")

                processed_games[name] = result
                new_results.append(result)

                # SAVE CHECKPOINT
                if len(new_results) % 50 == 0:
                    pd.DataFrame(processed_games.values()).to_csv(TEMP_CSV, index=False)

            elif response.status_code == 429:
                print("⏳ Rate Limit. Sleeping 5s...")
                time.sleep(5)
            else:
                print(f"💀 API Error {response.status_code}")

            time.sleep(0.26)  # Safe buffer for 4 req/s

        except Exception as e:
            print(f"⚠️ Exception for {name}: {e}")
            time.sleep(1)

    # 5. Final Merge
    print("\n💾 Saving final enriched dataset...")
    enrichment_df = pd.DataFrame(processed_games.values())

    # Merge with original
    final_df = pd.merge(df, enrichment_df, on='Name', how='left')

    # Fill NaNs
    cols_to_fix = ['Critic_Score', 'User_Score', 'Total_Rating_Count']
    final_df[cols_to_fix] = final_df[cols_to_fix].fillna(0)

    final_df.to_csv(OUTPUT_CSV, index=False)
    if os.path.exists(TEMP_CSV): os.remove(TEMP_CSV)

    print("-" * 40)
    print(f"🎉 DONE! Enriched data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    if CLIENT_ID == 'YOUR_CLIENT_ID_HERE':
        print("🛑 ERROR: Update keys in script!")
    else:
        run_enrichment()