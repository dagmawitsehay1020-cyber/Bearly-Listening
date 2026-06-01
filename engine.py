import pandas as pd
from thefuzz import process

print("Loading the Spotify dataset (1.08M songs)...")
df = pd.read_csv('tracks_features.csv')

df = df[(df['duration_ms'] >= 60000) & (df['duration_ms'] <= 600000)]
df = df.drop_duplicates(subset=['name', 'artists'])

df['clean_artists'] = df['artists'].str.replace(r"[\[\]']", "", regex=True)
df['search_text'] = df['name'] + " - " + df['clean_artists']

curated_top_songs = df.head(50)[['name', 'clean_artists']].values.tolist()

df['name_prefix'] = df['name'].str.lower().str[:2]
df['artist_prefix'] = df['clean_artists'].str.lower().str[:2]

print("Advanced Search & Pagination Systems Active!")

# === FUNCTION 1: PAGED BROWSING MECHANISM ===
def get_onboarding_page(songs, page_number):
    start_idx = page_number * 10
    end_idx = min(start_idx + 10, len(songs))

    page_songs = songs[start_idx:end_idx]

    print(f"\n --- ONBOARDING SHOWING PAGE {page_number + 1} (Songs {start_idx + 1} to {end_idx}) ---")

    for i, song in enumerate(page_songs, start_idx + 1):
        print(f"[{i}] {song[0]} by {song[1]}")
    print("----------------------------------------------")
    print("🔘 [ ⬅️ Previous 10 ]  |  🔘 [ ➡️ Next 10 ]  |  🔘 [ Done ✅ ]")

# === FUNCTION 2: FLEXIBLE FLEX SEARCH ENGINE ===
def flexible_search(user_query):
    print(f"\n Processing Search Input: '{user_query}'")
    query_clean = user_query.strip().lower()

    exact_match = df[
        (df['name'].str.lower() == query_clean) |
        (df['search_text'].str.lower() == query_clean) 
    ]

    if not exact_match.empty:
        perfect_matches = exact_match[['name', 'clean_artists']]
        print("Found Exact Match")
        for i, songs in enumerate(perfect_matches.values, 1):
            print(f"{i}. {songs}") 
        return perfect_matches.values.tolist()

    
    print("Exact Matches found for {user_query}")

    if "na -" in query_clean:
        artist_part = query_clean.split("na -", 1)[1].strip()
        matched_df = df[df['clean_artists'].str.lower().str.contains(artist_part, na=False)]

    else:
        if "-" in query_clean:
            title_part = query_clean.split("-", 1)[0].strip()
        else:
            title_part = query_clean

        prefix = title_part[:2]

        matched_df = df[df['name_prefix'] == prefix]

        if matched_df.empty or len(matched_df) < 10:
            matched_df = df[df['name_prefix'] == prefix[0]]

    if matched_df.empty:
        matched_df = df.head(1000)

    choices = matched_df['search_text'].tolist()
    best_matches = process.extract(user_query, choices, limit=20)
    print("Top Matches Found:")
    
    formatted_matches = []
    for i, match in enumerate(best_matches, 1):
        matched_text = match[0]
        score = match[1]
        if score >= 70:
            print(f" {i}. {matched_text} (Confidence score: {score}%)")

            origional_row = matched_df[matched_df['search_text'] == matched_text].iloc[0]
            formatted_matches.append([origional_row['name'], origional_row['clean_artists']])
    return formatted_matches

# get_onboarding_page(page_number=0)

# get_onboarding_page(page_number=1)

# flexible_search("NA - Rage Against The machine")

# flexible_search("Bohemian Rhapsody - Queen")

result = flexible_search("NA - Rage Against The machine")
get_onboarding_page(result, 1)


