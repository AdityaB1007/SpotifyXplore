from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import faiss
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        x = torch.cat([user_embed, item_embed], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x

app = FastAPI(title="Spotify Recommendation Engine")

print("Booting up Neural Engine...")

tracks_df = pd.read_pickle("processed_tracks.pkl")
unique_song_names = sorted(tracks_df['track_name'].unique())
unique_genres = sorted(tracks_df['genre'].dropna().unique().tolist())

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

item_encoder = encoders["item_encoder"]
user_encoder = encoders["user_encoder"]

index = faiss.read_index("spotify_vector_index.faiss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
model = NCFModel(num_users, num_items).to(device)

try:
    model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model weights could not be loaded. {e}")


class SongRequest(BaseModel):
    song_name: str
    k: int = 10

class PlaylistRequest(BaseModel):
    genres: List[str] = []
    energy_min: float = 0.0
    energy_max: float = 1.0
    dance_min: float = 0.0
    dance_max: float = 1.0
    tempo_min: int = 0
    tempo_max: int = 250
    limit: int = 50

@app.get("/")
def health_check():
    return {"status": "online", "tracks_loaded": len(tracks_df), "device": str(device)}

@app.get("/genres")
def get_genres():
    """Returns available genres for filtering."""
    return {"genres": unique_genres}

@app.get("/search")
def search_songs(q: str = Query(..., min_length=2), limit: int = 20):
    """
    Smart Search: Exact match > Starts with > Length > Alphabetical
    """
    q = q.lower()
    candidates = [song for song in unique_song_names if q in song.lower()]
    
    candidates.sort(key=lambda x: (
        x.lower() != q,
        not x.lower().startswith(q),
        len(x),
        x
    ))
    return {"matches": candidates[:limit]}

@app.post("/recommend/curate")
def curate_playlist(req: PlaylistRequest):
    """
    Filters the dataset based on audio features and returns a curated list.
    """
    df = tracks_df.copy()
   
    if req.genres:
        df = df[df['genre'].isin(req.genres)]
  
    df = df[
        (df['energy'] >= req.energy_min) & (df['energy'] <= req.energy_max) &
        (df['danceability'] >= req.dance_min) & (df['danceability'] <= req.dance_max) &
        (df['tempo'] >= req.tempo_min) & (df['tempo'] <= req.tempo_max)
    ]
    
    if 'popularity' in df.columns:
        df = df.sort_values(by='popularity', ascending=False)
    
    results = df.head(req.limit)
    
    playlist = []
    for _, row in results.iterrows():
        playlist.append({
            "name": row['track_name'],
            "artist": row['artist_name'],
            "genre": row['genre'],
            "year": int(row['year']),
            "energy": float(row['energy']),
            "danceability": float(row['danceability']),
            "tempo": float(row['tempo'])
        })
        
    return {
        "count": len(playlist),
        "playlist": playlist
    }

@app.post("/recommend/similar")
def recommend_similar(request: SongRequest):

    seed_row = tracks_df[tracks_df['track_name'] == request.song_name]
    if seed_row.empty:
        raise HTTPException(status_code=404, detail="Song not found")
    
    track_id = seed_row.iloc[0]['track_id']
  
    try:
        idx = item_encoder.transform([track_id])[0]
        with torch.no_grad():
            query_vector = model.item_embedding(torch.tensor(idx).to(device)).cpu().numpy()
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector extraction failed: {str(e)}")

    try:
        D, I = index.search(query_vector, request.k + 1)
        recommendations = []
        for rank, (score, idx) in enumerate(zip(D[0][1:], I[0][1:])):
            rec_track_id = item_encoder.inverse_transform([idx])[0]
            meta = tracks_df[tracks_df['track_id'] == rec_track_id].iloc[0]
            recommendations.append({
                "name": meta['track_name'],
                "artist": meta['artist_name'],
                "genre": meta['genre'],
                "year": int(meta['year']),
                "score": float(score)
            })
            
        return {"seed_song": request.song_name, "recommendations": recommendations}
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"FAISS search failed: {str(e)}")
