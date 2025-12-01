import os

# --- FIX FOR OPENMP ERROR #15 ---
# This must be set BEFORE importing torch or faiss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import faiss
import torch
import pickle
import numpy as np
import pandas as pd

# Load resources
print("Loading model and data...")
# Ensure these files exist in the same directory
tracks_df = pd.read_pickle("processed_tracks.pkl")

# Load model using CPU mapping to avoid CUDA errors during indexing if GPU is busy/unavailable
model_state = torch.load("ncf_model.pth", map_location="cpu")

# Extract Item Embeddings from the state dictionary
# The layer name depends on your NCFModel definition. 
# Based on your training script, it is 'item_embedding.weight'
if 'item_embedding.weight' in model_state:
    item_embeddings = model_state['item_embedding.weight'].numpy()
else:
    raise KeyError("Could not find 'item_embedding.weight' in model file. Check layer names.")

# FAISS Configuration
d = item_embeddings.shape[1] # Dimension (32)
# IndexFlatIP = Exact Inner Product (Cosine Similarity equivalent if normalized)
index = faiss.IndexFlatIP(d) 

print(f"Building Index for {item_embeddings.shape[0]} vectors...")

# Normalize vectors for Cosine Similarity
# This modifies the array in-place
faiss.normalize_L2(item_embeddings)

# Add vectors to the index
index.add(item_embeddings)

# Save the index to disk
output_file = "spotify_vector_index.faiss"
faiss.write_index(index, output_file)
print(f"Success! Vector Index built and saved to {output_file}")