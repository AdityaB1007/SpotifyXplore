import faiss
import torch
import pickle
import numpy as np
import pandas as pd

print("Loading model and data...")
tracks_df = pd.read_pickle("processed_tracks.pkl")

model_state = torch.load("ncf_model.pth", map_location="cpu")

if 'item_embedding.weight' in model_state:
    item_embeddings = model_state['item_embedding.weight'].numpy()
else:
    raise KeyError("Could not find 'item_embedding.weight' in model file. Check layer names.")

d = item_embeddings.shape[1]
index = faiss.IndexFlatIP(d) 

print(f"Building Index for {item_embeddings.shape[0]} vectors...")

faiss.normalize_L2(item_embeddings)

index.add(item_embeddings)

output_file = "spotify_vector_index.faiss"
faiss.write_index(index, output_file)

print(f"Success! Vector Index built and saved to {output_file}")
