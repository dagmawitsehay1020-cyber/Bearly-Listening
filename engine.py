import os
import json
import pandas as pd
import numpy as np
import IPython
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ==================================================================================================
# === 1. CORE FUNCTIONS ============================================================================
# ==================================================================================================

_df = None
_titles_dict = {}
_artists_dict = {}
_combined_dict = {}
_name_lower = None
_artist_lower = None
_combined_search = None

_partitioned_titles = defaultdict(dict)
_partitioned_artists = defaultdict(dict)
_partitioned_combined = defaultdict(dict)

def init_search_engine(dataframe: pd.DataFrame):
    global _df, _name_lower, _artist_lower, _combined_search
    global _partitioned_titles, _partitioned_artists, _partitioned_combined
    
    print("⚡ Mapping pre-baked Parquet search matrices into alphabet partitions...")
    _df = dataframe
    
    _name_lower = _df['name_lower'].fillna("").astype(str)
    _artist_lower = _df['artist_lower'].fillna("").astype(str)
    _combined_search = _df['combined_search'].fillna("").astype(str)
    
    _partitioned_titles.clear()
    _partitioned_artists.clear()
    _partitioned_combined.clear()

    for idx, (name, artist, combined) in enumerate(zip(_name_lower.tolist(), _artist_lower.tolist(), _combined_search.tolist())):
        
        first_char_name = name[0] if (isinstance(name, str) and name) else ''
        first_char_artist = artist[0] if (isinstance(artist, str) and artist) else ''
        
        if first_char_name:
            _partitioned_titles[first_char_name][idx] = name
        if first_char_artist:
            _partitioned_artists[first_char_artist][idx] = artist
            
        if first_char_name:
            _partitioned_combined[first_char_name][idx] = combined
        if first_char_artist and first_char_artist != first_char_name:
            _partitioned_combined[first_char_artist][idx] = combined

    print(f"✅ Engine state armed. {_df.shape[0]:,} tracks sorted into sub-matrix buckets.")

def get_ui_song_list(songs, page_number):
    start_idx = page_number * 10
    end_idx = min(start_idx + 10, len(songs))
    page_songs = songs[start_idx:end_idx]

    ui_buffer = [f"\n --- SONG LIST {page_number + 1} (Songs {start_idx + 1} to {end_idx}) ---"]
    for i, song in enumerate(page_songs, start_idx + 1):
        ui_buffer.append(f"[{i}] {song[0]} by {song[1]}")
    ui_buffer.append("----------------------------------------------")
    ui_buffer.append("🔘 [ ⬅️ Previous 10 ]  |  🔘 [ ➡️ Next 10 ]  |  🔘 [ Done ✅ ]")
    print("\n".join(ui_buffer))

def flexible_two_input_search(query_string) -> list:
    if _df is None or _df.empty:
        raise RuntimeError("Search engine has not been initialized.")

    query_clean = query_string.lower().strip()
    if not query_clean:
        return []

    first_char = query_clean[0]

    delimiter = None
    for d in [" - ", " by ", "-", "by"]:
        if d in query_clean:
            delimiter = d
            break

    if delimiter:
        parts = query_clean.split(delimiter, 1)
        part1, part2 = parts[0].strip(), parts[1].strip()

        if part1 and part2:
            mask1 = (_name_lower.str.contains(part1, na=False, regex=False) & _artist_lower.str.contains(part2, na=False, regex=False))
            mask2 = (_name_lower.str.contains(part2, na=False, regex=False) & _artist_lower.str.contains(part1, na=False, regex=False))
            
            fast_df = _df[mask1 | mask2]
            if not fast_df.empty:
                return fast_df[['name', 'clean_artists']].head(30).values.tolist()

            p1_char, p2_char = part1[0], part2[0]
            t_dict = _partitioned_titles.get(p1_char, {})
            a_dict = _partitioned_artists.get(p2_char, {})
            
            t_dict_rev = _partitioned_titles.get(p2_char, {})
            a_dict_rev = _partitioned_artists.get(p1_char, {})

            matched_title = process.extract(part1, t_dict, limit=15) if t_dict else []
            matched_artist = process.extract(part2, a_dict, limit=15) if a_dict else []
            
            matched_title_rev = process.extract(part2, t_dict_rev, limit=15) if t_dict_rev else []
            matched_artist_rev = process.extract(part1, a_dict_rev, limit=15) if a_dict_rev else []

            valid_indices = set()
            if matched_title and matched_artist:
                valid_indices.update({m[2] for m in matched_title if m[1] >= 75}.intersection({m[2] for m in matched_artist if m[1] >= 75}))
            if matched_title_rev and matched_artist_rev:
                valid_indices.update({m[2] for m in matched_title_rev if m[1] >= 75}.intersection({m[2] for m in matched_artist_rev if m[1] >= 75}))

            if valid_indices:
                return _df.iloc[list(valid_indices)][['name', 'clean_artists']].head(30).values.tolist()

    fast_fallback_df = _df[_combined_search.str.contains(query_clean, na=False, regex=False)]
    if len(fast_fallback_df) >= 10:
        return fast_fallback_df[['name', 'clean_artists']].head(30).values.tolist()

    c_dict = _partitioned_combined.get(first_char, {})
    if not c_dict:
        return []

    raw_matches = process.extract(query_clean, c_dict, limit=20)

    processed_results = []
    for match in raw_matches:
        matched_str, score, index = match[0], match[1], match[2]
        row = _df.iloc[index]
        song_name, artist_name = row['name'], row['clean_artists']

        boost = 0
        if query_clean in song_name.lower() or query_clean in artist_name.lower():
            boost = 35

        processed_results.append((score + boost, [song_name, artist_name]))

    processed_results.sort(key=lambda x: x[0], reverse=True)
    return [track_data for _, track_data in processed_results[:30]]


def calculate_taste_vector(user_ratings, df):
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    working_df = df.copy()
    working_df['loudness'] = (working_df['loudness'] - (-60.0)) / (0.0 - (-60.0))
    working_df['tempo'] = (working_df['tempo'] - (0.0)) / (250.0 - (0.0))
    working_df[features] = np.clip(working_df[features], 0.0, 1.0)

    working_df['lookup_key'] = working_df.name.str.lower() + ' - ' + working_df.clean_artists.str.lower() 

    keys = working_df['lookup_key'].to_numpy()
    vectors = working_df[features].to_numpy(dtype=float)
    working_dict = dict(zip(keys, vectors))

    rating_weight_map = {5: 2.5, 4: 1.5, 3: 0.2, 2: -1.5, 1: -2.5}
    weighted_features_list = []
    total_weight = 0

    for song_name, artist_name, rating in user_ratings:
        search_string = song_name.lower() + ' - ' + artist_name.lower()
        if search_string not in working_dict:
            continue
        raw_vector = working_dict[search_string].copy()
        weight = rating_weight_map[rating]

        if weight >= 0:
            weighted_vector = raw_vector * weight
        else:
            weighted_vector = (1.0 - raw_vector) * abs(weight)

        weighted_features_list.append(weighted_vector)
        total_weight += abs(weight)

    if not weighted_features_list or total_weight == 0:
        return np.zeros(9)
    
    taste_vector = np.sum(weighted_features_list, axis=0) / total_weight
    return np.clip(taste_vector, 0.0, 1.0)

def generate_geo_recommendations(taste_vector, dataframe, region="global", exclude_tracks=None):
    """
    Filters the music matrix by geographical clusters, drops already-rated songs, 
    normalizes unscaled features on-the-fly, and calculates cosine similarity.
    """
    GEO_GROUPS = {
        "african": ["African", "Caribbean"],
        "european": ["UK", "Baltic / Europe", "Scandinavian", "German", "Spanish", "French", "Italian", "Dutch", "Irish", "Scottish", "Portuguese", "Romanian", "Polish", "Austrian", "Belgian"],
        "asian": ["Korean", "Japanese", "Taiwan", "Chinese", "Indonesia", "Malaysian", "Indian", "Arabic", "Russian", "Israeli", "Mongolian"],
        "anglo_americas": ["Canadian", "Australian"],
        "global": ["Global"]
    }
    
    working_pool = dataframe.copy()
    if region != "global" and region in GEO_GROUPS:
        allowed_regions = GEO_GROUPS[region]
        working_pool = working_pool[
            working_pool['region_1'].isin(allowed_regions) |
            working_pool['region_2'].isin(allowed_regions) |
            working_pool['region_3'].isin(allowed_regions)
        ]
    
    print(f"🧮 Calculating similarities across {len(working_pool):,} available tracks...")

    if exclude_tracks:
        pool_signatures = working_pool['name'].str.lower() + "|||" + working_pool['clean_artists'].str.lower()
        exclude_signatures = {f"{t[0].lower()}|||{t[1].lower()}" for t in exclude_tracks}
        working_pool = working_pool[~pool_signatures.isin(exclude_signatures)]

    if working_pool.empty:
        print("⚠️ Warning: Filtered pool is completely empty. Falling back to baseline dataframe.")
        working_pool = dataframe.copy()

    sample_size = min(30000, len(working_pool)) 
    random_sample = working_pool.sample(n=sample_size).copy()
    
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    feature_matrix = random_sample[audio_features].values.copy()

    feature_matrix[:, 2] = (feature_matrix[:, 2] - (-60.0)) / (0.0 - (-60.0))
    feature_matrix[:, 8] = (feature_matrix[:, 8] - 0.0) / (250.0 - 0.0)
    feature_matrix = np.clip(feature_matrix, 0.0, 1.0)
    
    target_vector = np.array(taste_vector).reshape(1, -1)
    
    random_sample['cos_sim'] = cosine_similarity(target_vector, feature_matrix).flatten()
    random_sample.sort_values('cos_sim', ascending=False, inplace=True)

    recommendations = []
    for row in random_sample.itertuples():
        if len(recommendations) >= 50:
            break
        if row.cos_sim >= 0.9999:
            continue
        recommendations.append([row.name, row.clean_artists])

    if len(recommendations) == 0:
        for row in random_sample.head(10).itertuples():
            recommendations.append([row.name, row.clean_artists])
    
    return recommendations

# ==================================================================================================
# === 2. STORAGE LAYER =============================================================================
# ==================================================================================================

def load_interacted_songs_from_disk(history_file_path):
    if not history_file_path or not os.path.exists(history_file_path):
        return []
    with open(history_file_path, 'r') as file:
        return json.load(file)


def save_interacted_songs_to_disk(current_session_selections, history_file_path):
    past_song_list = load_interacted_songs_from_disk(history_file_path)
    for track in current_session_selections: 
        track_info = [track[0], track[1]]
        if track_info not in past_song_list:
            past_song_list.append(track_info)
    with open(history_file_path, 'w') as file:
        json.dump(past_song_list, file, indent=4)


def initialize_onboarding_tracks(master_df, history_file_path):
    past_songs_list = load_interacted_songs_from_disk(history_file_path) if os.path.exists(history_file_path) else []

    if len(past_songs_list) > 0:
        print("👋 Welcome Back to Bearly Listening!")
        print("[1] Load my previously rated songs onto the landing page")
        print("[2] Give me a fresh, random selection of songs")
        if input("Press 1 or 2: ").strip() == "1":
            return past_songs_list
    
    return master_df.sample(n=50)[['name', 'clean_artists']].values.tolist()

# ==================================================================================================
# === 3. TERMINAL UI LOOP ==========================================================================
# ==================================================================================================

def run_interaction_loop(master_songs_list, master_df, history_file_path):
    current_page = 0
    user_selections = []
    active_view_tracks = master_songs_list
    current_mode = "browse"
    
    while True:
        IPython.display.clear_output(wait=True)

        if current_mode == 'browse':
            print("🎵 --- BASE ONBOARDING TRACK CATALOG ---")
            get_ui_song_list(active_view_tracks, current_page)
        elif current_mode == 'search':
            print("🔍 Active Search Results Matrix")
            get_ui_song_list(active_view_tracks, current_page)            
            print("Options: ['back' to escape search mode] or ['done' to view final picks]")
        elif current_mode == 'recommendations':
            print("✨ --- YOUR PERSONALIZED RECOMMENDATIONS SPECTRUM ---")
            get_ui_song_list(active_view_tracks, current_page)
            print("Options: ['next'/'prev' to flip pages] | ['exit' to close application]")

        print(f"🛒 Current Logged Selections: {len(user_selections)} tracks registered.")
        
        if current_mode == 'recommendations':
            user_input = input("📝 Enter 'next'/'prev' to browse matches, or 'exit' to quit: ").lower().strip()
        else:
            user_input = input("📝 Enter item # to rate, 'search', 'next'/'prev', 'list', 'restart' or 'done': ").lower().strip()

        if user_input == 'search':
            IPython.display.clear_output(wait=True)
            search_results = flexible_two_input_search(master_df)
            if search_results:
                active_view_tracks = search_results
                current_mode = 'search'
            else:
                active_view_tracks = master_songs_list
                current_mode = 'browse'
            current_page = 0

        elif user_input == 'back':
            active_view_tracks = master_songs_list
            current_mode = 'browse'
            current_page = 0
        
        elif user_input == 'next':
            max_page_idx = (len(active_view_tracks) - 1) // 10
            if current_page < max_page_idx:
                current_page += 1

        elif user_input == 'prev':
            if current_page > 0:
                current_page -= 1

        elif user_input == 'list':
            if user_selections:
                print("📋 === Your Logged Preference Profile ===")
                for i, item in enumerate(user_selections):
                    print(f"[{i+1}]. {item[0]} by {item[1]} -> {item[2]}⭐")
                input("Press Enter to return...")
            else:
                input("⚠️ Profile is empty! Press Enter...")
            
        elif user_input == 'restart':
            user_selections = []
            active_view_tracks = master_songs_list
            current_mode = 'browse'
            current_page = 0
                
        elif user_input.isdigit():
            target_idx = int(user_input)
            is_valid = False

            if current_mode == 'browse':
                start_bound = (current_page * 10) + 1
                end_bound = min((start_bound + 9), len(active_view_tracks))
                is_valid = (start_bound <= target_idx <= end_bound)
            else:
                is_valid = (1 <= target_idx <= len(active_view_tracks))

            if is_valid: 
                found_song = active_view_tracks[target_idx - 1]
                rating_input = input(f"⭐ Rate {found_song[0]} from 1 to 5 stars: ")

                if rating_input.isdigit() and int(rating_input) in range(1, 6):
                    user_selections = [x for x in user_selections if not (x[0] == found_song[0] and x[1] == found_song[1])]
                    user_selections.append([found_song[0], found_song[1], int(rating_input)])
                    
                    if current_mode == 'search':
                        search_results = flexible_two_input_search(master_df)
                        if search_results:
                            active_view_tracks = search_results
                        else:
                            active_view_tracks = master_songs_list
                            current_mode = 'browse'
                        current_page = 0
    
        elif user_input == 'done':
            if not user_selections:
                break

            IPython.display.clear_output(wait=True)
            
            # --- 1. CALCULATE TASTE SPECTRUM ---
            features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            taste_vector = calculate_taste_vector(user_selections, master_df)
            
            # --- 2. PRINT SESSION PROFILE SUMMARY ---
            print("\n📊 === YOUR SELECTION AUDIT ===")
            rating_weight_map = {5: 2.5, 4: 1.5, 3: 0.2, 2: -1.5, 1: -2.5}
            
            for name, artist, rating in user_selections:
                weight = rating_weight_map[rating]
                direction = "🔺 Boosts" if weight > 0 else "🔻 Suppresses"
                print(f" • '{name}' by {artist} -> {rating}⭐ ({direction} these traits by {abs(weight)}x)")
            
            # --- 3. PRINT AUDIO TRAIT SPECTRUM MATRIX ---
            print("\n📐 === TARGET VIBE VECTOR COMPONENTS ===")
            print(f"{'Audio Feature':<20} | {'Target Value (0.0 - 1.0)':<25} | Vibe Description")
            print("-" * 75)
            
            descriptions = {
                'danceability': "Rhythmic stability, tempo cadence, and beat strength.",
                'energy': "Perceived intensity, activity level, and sonic roar.",
                'loudness': "Overall signal amplitude strength (normalized).",
                'speechiness': "Presence of spoken words vs. melodic tracks.",
                'acousticness': "Probability that the track uses non-electronic instruments.",
                'instrumentalness': "Likelihood of a track containing zero vocal presence.",
                'liveness': "Auditory cues of an active audience or venue setting.",
                'valence': "Musical positiveness (High = Happy/Upbeat, Low = Dark/Melancholy).",
                'tempo': "Overall pace of the track rhythm (normalized)."
            }
            
            for idx, feature in enumerate(features):
                val = taste_vector[idx]
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                print(f"{feature:<20} | {val:.4f}  [{bar}] | {descriptions[feature]}")
                
            print("-" * 75)
            input("\n✨ Vibe diagnostics complete. Press Enter to select geographic filters...")

            # --- 4. GEOGRAPHIC FILTERS ---
            IPython.display.clear_output(wait=True)
            print("🌐 --- RECOMMENDATION GEOGRAPHIC FILTER ---")
            available_regions = sorted([str(r) for r in master_df['region_1'].unique() if pd.notna(r) and r != "Global"])
            
            for idx, region in enumerate(available_regions, start=1):
                print(f"[{idx}] {region}")
            global_option_idx = len(available_regions) + 1
            print(f"[{global_option_idx}] No Region Filter")

            raw_geo_input = input(f"\nSelect option(s) separated by commas: ").strip()
            input_choices = [c.strip() for c in raw_geo_input.split(",") if c.strip()]

            target_regions = []
            for choice in input_choices:
                if choice.isdigit():
                    ch_idx = int(choice) - 1
                    if 0 <= ch_idx < len(available_regions):
                        target_regions.append(available_regions[ch_idx])
                    elif int(choice) == global_option_idx:
                        target_regions = []
                        break
            
            target_regions = None if not target_regions else target_regions

            print("\n🎯 --- RECOMMENDATION FILTER CONFIGURATION ---")
            print("[1] Include history database tracks.\n[2] Discovery mode only.")
            filter_choice = input("Select option (1 or 2): ").strip()

            exclude_set = []
            if filter_choice == "2":
                historical_pool = load_interacted_songs_from_disk(history_file_path)
                exclude_set.extend([[item[0], item[1]] for item in historical_pool])
                exclude_set.extend([[item[0], item[1]] for item in user_selections])

            recommendation_pool = generate_geo_recommendations(
                test_vector=taste_vector, 
                master_df=master_df, 
                exclude_tracks=exclude_set, 
                target_region=target_regions
            )
            active_view_tracks = recommendation_pool
            current_mode = 'recommendations'
            current_page = 0
            
        elif user_input == 'exit' and current_mode == 'recommendations':
            save_interacted_songs_to_disk(user_selections, history_file_path)
            break

# ==================================================================================================
# === 4. DATA INITIALIZATION & ENTRYPOINT =========================================================
# ==================================================================================================

if os.path.exists("tracks_features.parquet"):
    df = pd.read_parquet("tracks_features.parquet", engine="pyarrow")
else:
    df = pd.read_csv('tracks_features.csv')
    df = df[(df['duration_ms'] >= 60000) & (df['duration_ms'] <= 600000)].drop_duplicates(subset=['name', 'artists'])
    df['clean_artists'] = df['artists'].str.replace(r"[\[\]']", "", regex=True)
    df['search_text'] = df['name'] + " - " + df['clean_artists']
    df['name_prefix'] = df['name'].str.lower().str[:2]
    df['artist_prefix'] = df['clean_artists'].str.lower().str[:2]

# Keep this completely empty or delete it so the terminal UI loop doesn't fight the bot
if __name__ == "__main__":
    print("Engine module loaded successfully.")