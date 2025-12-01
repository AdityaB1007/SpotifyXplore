import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
import pickle
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_FILE = r"spotify_1mill_dataset\spotify_data.csv" 
NUM_USERS = 100    
EMBEDDING_DIM = 32           
BATCH_SIZE = 256               
EPOCHS = 10

def load_and_process_data(filepath):
    """
    Loads the real Spotify dataset and prepares it for the model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Please upload your dataset.")

    print(f"Loading data from {filepath}...")
    
    try:
        tracks_df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None

    required_cols = [
        'track_id', 'artist_name', 'track_name', 'popularity', 'year', 'genre',
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'time_signature'
    ]
    
    existing_cols = [c for c in required_cols if c in tracks_df.columns]
    tracks_df = tracks_df[existing_cols].copy()
    
    tracks_df.dropna(inplace=True)
    
    tracks_df.drop_duplicates(subset=['track_id'], inplace=True)
   

    print(f"Data Loaded: {len(tracks_df)} unique tracks.")

    print("Scaling audio features...")
    scaler = MinMaxScaler()
    
    scale_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                  'instrumentalness', 'liveness', 'valence', 'tempo']
    
    for col in scale_cols:
        if col in tracks_df.columns:
            tracks_df[col] = pd.to_numeric(tracks_df[col], errors='coerce')
    
    tracks_df.dropna(subset=scale_cols, inplace=True)

    scaled_data = scaler.fit_transform(tracks_df[scale_cols])
    scaled_cols_names = [f'scaled_{col}' for col in scale_cols]
    tracks_df[scaled_cols_names] = scaled_data

    print("Clustering songs based on audio features...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    tracks_df['cluster_label'] = kmeans.fit_predict(tracks_df[scaled_cols_names])
 
    print("Simulating user listening history for NCF training...")
    interactions = []
    
    all_track_ids = tracks_df['track_id'].values
    
    for user_id in range(NUM_USERS):
        
        num_listened = np.random.randint(20, 50)
        
        preferred_cluster = np.random.choice(range(8))
        
        cluster_songs = tracks_df[tracks_df['cluster_label'] == preferred_cluster]['track_id'].values
        other_songs = tracks_df[tracks_df['cluster_label'] != preferred_cluster]['track_id'].values
        
        listened_tracks = []
        if len(cluster_songs) > 0:
             count_pref = min(len(cluster_songs), int(num_listened * 0.7))
             listened_tracks.extend(np.random.choice(cluster_songs, count_pref, replace=False))
        
        needed = num_listened - len(listened_tracks)
        if needed > 0 and len(other_songs) > 0:
            listened_tracks.extend(np.random.choice(other_songs, needed, replace=False))
            
        for track in listened_tracks:
            interactions.append({'user_id': user_id, 'track_id': track, 'interaction': 1.0})
            
    interaction_df = pd.DataFrame(interactions)
    
    return tracks_df, interaction_df, scaled_cols_names

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

class SpotifyDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

def train_pipeline():
    tracks_df, interactions_df, scaled_cols = load_and_process_data(DATA_FILE)
    
    if tracks_df is None:
        return 
    
    print("Encoding User and Item IDs...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    interactions_df['user_idx'] = user_encoder.fit_transform(interactions_df['user_id'])
    
    item_encoder.fit(tracks_df['track_id'])
    interactions_df['item_idx'] = item_encoder.transform(interactions_df['track_id'])

    train_data = SpotifyDataset(
        interactions_df['user_idx'].values,
        interactions_df['item_idx'].values,
        interactions_df['interaction'].values
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    num_users = interactions_df['user_idx'].nunique()
    num_items = len(item_encoder.classes_)
    
    print(f"Initializing Model for {num_users} users and {num_items} items...")
    model = NCFModel(num_users, num_items, EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"Starting training on {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for user_batch, item_batch, rating_batch in train_loader:
            user_batch, item_batch, rating_batch = user_batch.to(DEVICE), item_batch.to(DEVICE), rating_batch.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(user_batch, item_batch).squeeze()
            loss = criterion(predictions, rating_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    print("Saving model and data...")
    torch.save(model.state_dict(), "ncf_model.pth")
    tracks_df.to_pickle("processed_tracks.pkl")
    
    with open("encoders.pkl", "wb") as f:
        pickle.dump({
            "user_encoder": user_encoder, 
            "item_encoder": item_encoder,
            "scaled_cols": scaled_cols
        }, f)

    print("Training complete. Files saved: ncf_model.pth, processed_tracks.pkl, encoders.pkl")

if __name__ == "__main__":
    train_pipeline()
