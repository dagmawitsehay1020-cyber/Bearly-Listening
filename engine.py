import pandas as pd
import numpy as np
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

class MusicEngine:
    def __init__(self, file_paths):
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]

    def _get_data(self, columns=None):
        dfs = [pd.read_parquet(f, engine="pyarrow", columns=columns) for f in self.file_paths]
        return pd.concat(dfs, ignore_index=True)

    def get_geo_counts(self):
        """Calculates region counts on the fly."""
        df = self._get_data(columns=['region_1', 'region_2', 'region_3'])
        all_regions = pd.concat([df['region_1'], df['region_2'], df['region_3']]).dropna()
        return all_regions.value_counts().to_dict()

    def flexible_two_input_search(self, query_string):
        df = self._get_data(columns=['name', 'clean_artists'])
        query_clean = query_string.lower().strip()
        mask = (df['name'].str.contains(query_clean, na=False, case=False) | 
                df['clean_artists'].str.contains(query_clean, na=False, case=False))
        return df[mask].head(30).values.tolist()

    def calculate_taste_vector(self, user_ratings):
        df = self._get_data(columns=['name', 'clean_artists'] + self.audio_features)
        rating_weight_map = {5: 2.5, 4: 1.5, 3: 0.2, 2: -1.5, 1: -2.5}
        weighted_vectors, total_weight = [], 0

        for song_name, artist_name, rating in user_ratings:
            match = df[(df['name'].str.lower() == song_name.lower()) & 
                       (df['clean_artists'].str.lower() == artist_name.lower())]
            if not match.empty:
                raw_v = match[self.audio_features].values[0].astype(float)
                # Normalize
                raw_v[2] = (raw_v[2] + 60) / 60 
                raw_v[8] = raw_v[8] / 250
                weight = rating_weight_map[rating]
                weighted_vectors.append(raw_v * weight if weight >= 0 else (1.0 - raw_v) * abs(weight))
                total_weight += abs(weight)
        return np.clip(np.sum(weighted_vectors, axis=0) / total_weight, 0.0, 1.0) if total_weight > 0 else np.zeros(9)

    def generate_geo_recommendations(self, taste_vector, region="global", exclude_tracks=None):
        GEO_GROUPS = {
            "african": ["African", "Caribbean"],
            "european": ["UK", "Baltic / Europe", "Scandinavian", "German", "Spanish", "French", "Italian", "Dutch", "Irish", "Scottish", "Portuguese", "Romanian", "Polish", "Austrian", "Belgian"],
            "asian": ["Korean", "Japanese", "Taiwan", "Chinese", "Indonesia", "Malaysian", "Indian", "Arabic", "Russian", "Israeli", "Mongolian"],
            "anglo_americas": ["Canadian", "Australian"],
            "global": ["Global"]
        }
        df = self._get_data(columns=self.audio_features + ['name', 'clean_artists', 'region_1', 'region_2', 'region_3'])
        
        if region != "global":
            allowed = GEO_GROUPS.get(region, [])
            df = df[df[['region_1', 'region_2', 'region_3']].isin(allowed).any(axis=1)]
            
        feature_matrix = df[self.audio_features].values
        feature_matrix[:, 2] = (feature_matrix[:, 2] + 60) / 60
        feature_matrix[:, 8] = feature_matrix[:, 8] / 250
        
        sims = cosine_similarity(taste_vector.reshape(1, -1), np.clip(feature_matrix, 0, 1)).flatten()
        df = df.assign(cos_sim=sims).sort_values('cos_sim', ascending=False)
        return df[['name', 'clean_artists']].head(50).values.tolist()

engine = MusicEngine(['tracks_features_part1.parquet', 'tracks_features_part2.parquet'])