import pandas as pd
import numpy as np

print("💿 Loading raw CSV and applying filters...")
df = pd.read_csv('tracks_features.csv')

df = df[(df['duration_ms'] >= 60000) & (df['duration_ms'] <= 600000)]
df = df.drop_duplicates(subset=['name', 'artists'])

print("✨ Engineering optimized identity and search columns...")
df['clean_artists'] = df['artists'].str.replace(r"[\[\]']", "", regex=True)

df['combined_search'] = (
    df['name'].astype(str).str.lower() + " " + 
    df['clean_artists'].astype(str).str.lower()
)

geo_lookup_matrix = {
    "Korean": ["5th gen k-pop", "classic k-pop", "classic korean pop", "joseon pop", "k-*", "korean*", "korea*"], 
    "Afghanistan": ["afghan pop"], 
    "African": ["african rock", "afrikaans*", "afro*", "cameroonian pop", "cape town indie", "ghanaian*", "kenyan*", "malawian*", "malian blues", "naija worship", "nigerian*", "south african*", "sudanese*", "tanzanian*"], 
    "Japanese": ["anime*", "j-*", "japanese*"], 
    "Arabic": ["arab*", "classic arab pop", "classic persian pop", "lebanese pop", "libyan pop", "egyptian*", "moroccan pop"], 
    "Argentine": ["argentine*"], 
    "Australian": ["aussie drill", "auckland indie", "australian*", "brisbane*", "melbourne*"], 
    "Austrian": ["austrian*", "austro-german modernism", "austropop"],
    "Indonesia": ["bali indie", "indonesian*"], 
    "Caribbean": ["barbadian pop", "jamaican*"],
    "Belgian": ["belgian*"], 
    "German": ["bergen indie", "frankfurt electronic", "german*", "neue deutsche harte", "neue neue deutsche welle"], 
    "UK": ["birmingham*", "brighton indie", "bristol*", "british*", "britpop", "britpop revival", "cambridgeshire indie", "cardiff indie", "celtic rock", "classic uk pop" ,"east anglia indie", "edmonton indie", "leeds indie", "leicester indie", "liverpool indie", "london*", "manchester*", "northern irish indie", "north east england indie", "nottingham indie", "uk *"], 
    "Scottish": ["scottish*"], 
    "Irish": ["dublin indie", "irish*"],
    "Portuguese": ["brazilian*", "portuguese*"], 
    "Baltic / Europe": ["bulgarian r&b", "czech pop", "eurodance", "europop", "euroska", "hungarian*", "moldovan pop"], 
    "Canadian": ["canadian*", "classic canadian rock"], 
    "Chinese": ["chinese*", "hong kong indie"], 
    "Scandinavian": ["classic danish pop", "danish*", "classic swedish pop", "finnish*", "icelandic*", "nordic*", "norwegian*", "scandinavian r&b", "scandipop", "swedish*", "viking metal"], 
    "Italian": ["classic italian pop", "italian*"], 
    "Spanish": ["colombian indie", "colombian pop", "dominican indie", "drill espanol", "el paso indie", "flamenco*", "guyanese pop", "latin*", "latintronica", "latinx alternative", "mexican*", "nuevo*", "pop venezolano", "ska mexicano", "spanish*", "trap latino"],         
    "Indian": ["desi*", "gujarati pop", "hindi indie", "indian*", "new delhi indie"], 
    "French": ["drill francais", "french*", "frenchcore"], 
    "Dutch": ["dutch*"], 
    "Ethiopian": ["ethio-jazz"],         
    "Greek": ["greek*"], 
    "Israeli": ["israeli*"], 
    "Malaysian": ["malaysian*"], 
    "Mongolian": ["mongolian alternative"], 
    "Polish": ["polish*"], 
    "Romanian": ["romanian*"], 
    "Russian": ["russian*"], 
    "Singaporean": ["singaporean"], 
    "Swiss": ["swiss*"], 
    "Taiwan": ["taiwan*"], 
    "Thai": ["thai*"], 
    "Turkish": ["turkish*"], 
    "Ukranian": ["ukranian*"], 
    "Vietnamese": ["viet*"], 
    "Welsh": ["welsh"], 
}

explicit_lookup = {}
wildcard_lookup = []

for region, patterns in geo_lookup_matrix.items():
    for pattern in patterns:
        p_clean = pattern.lower().strip()
        if p_clean.endswith('*'):
            prefix_token = p_clean[:-1].strip()
            wildcard_lookup.append((prefix_token, region))
        else:
            explicit_lookup[p_clean] = region

wildcard_lookup.sort(key=lambda x: len(x[0]), reverse=True)


def find_region_for_genre(genre):
    genre = genre.lower().strip()
    # 1. Exact Match Check (O(1))
    if genre in explicit_lookup:
        return explicit_lookup[genre]
    # 2. Sequential Suffix Check
    for prefix, region in wildcard_lookup:
        if genre.startswith(prefix):
            return region
    return "Global"


def extract_geo_columns(genre_string):
    if pd.isna(genre_string):
        return [None, None, None]
    genre_str_clean = str(genre_string).strip()
    if not genre_str_clean:
        return [None, None, None]
    
    individual_genres = [g.strip() for g in genre_str_clean.split(",")]
    found_region = {}
    for g in individual_genres:
        if not g:
            continue
        region = find_region_for_genre(g)
        found_region[region] = found_region.get(region, 0) + 1

    sorted_region = sorted(found_region.keys(), key=lambda r: (-found_region[r], r))
    while len(sorted_region) < 3:
        sorted_region.append(None)
    return sorted_region[:3]


print("🌐 Re-building dynamic dictionary maps from artist profiles...")
region_df = pd.read_csv("spotify_artist_info.csv")[['names', 'genres']]
region_df['region_list'] = region_df['genres'].apply(extract_geo_columns)

region_df['match_key'] = region_df['names'].astype(str).str.lower().str.strip()
region_df = region_df.drop_duplicates(subset=['match_key'])

artist_to_geo1 = dict(zip(region_df['match_key'], region_df['region_list'].str[0]))
artist_to_geo2 = dict(zip(region_df['match_key'], region_df['region_list'].str[1]))
artist_to_geo3 = dict(zip(region_df['match_key'], region_df['region_list'].str[2]))


print("🗺️ Mapping geographical metadata onto loaded dataframe index...")
track_artist_keys = df['clean_artists'].astype(str).str.lower().str.strip()

df['region_1'] = track_artist_keys.map(artist_to_geo1).fillna("Global")
df['region_2'] = track_artist_keys.map(artist_to_geo2)
df['region_3'] = track_artist_keys.map(artist_to_geo3)

essential_columns = [
    'name', 
    'clean_artists',
    'combined_search',
    'region_1', 
    'region_2', 
    'region_3',
    'danceability', 
    'energy', 
    'loudness', 
    'speechiness', 
    'acousticness', 
    'instrumentalness', 
    'liveness', 
    'valence', 
    'tempo'
]

print(f"📦 Filtering dataset matrix down to {len(essential_columns)} essential columns...")
optimized_df = df[essential_columns].copy()

float_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
optimized_df[float_cols] = optimized_df[float_cols].astype('float32')

print("\n🔍 Verification Snapshot (First 5 rows):")
print(optimized_df[['name', 'clean_artists', 'region_1', 'region_2']].head())

print("\n🚀 Overwriting tracks_features.parquet with optimized, lightweight footprints...")

print("\n🚀 Splitting and saving tracks_features into lightweight cloud chunks...")

midpoint = len(optimized_df) // 2

df_part1 = optimized_df.iloc[:midpoint].copy()
df_part2 = optimized_df.iloc[midpoint:].copy()

df_part1.to_parquet('tracks_features_part1.parquet', engine='pyarrow', compression='snappy')
df_part2.to_parquet('tracks_features_part2.parquet', engine='pyarrow', compression='snappy')

print("Done! Created tracks_features_part1.parquet and tracks_features_part2.parquet.")