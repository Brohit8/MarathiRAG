"""
Generate embeddings for the dictionary.
Run this ONCE locally before deploying.

Usage:
    python generate_embeddings.py

This creates embeddings.npy which you'll upload with your app.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DICTIONARY_PATH = "berntsen_dictionary_processed.json"
OUTPUT_PATH = "embeddings.npy"
MODEL_NAME = "l3cube-pune/marathi-sentence-similarity-sbert"
BATCH_SIZE = 64

def main():
    # Load dictionary
    print("📖 Loading dictionary...")
    with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    print(f"   Loaded {len(dictionary):,} entries")
    
    # Load model
    print("🧠 Loading MahaSBERT model...")
    model = SentenceTransformer(MODEL_NAME)
    print("   Model loaded!")
    
    # Extract search texts
    search_texts = [entry['search_text'] for entry in dictionary]
    
    # Generate embeddings in batches
    print("🔄 Generating embeddings...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(search_texts), BATCH_SIZE), desc="Batches"):
        batch = search_texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings)
    print(f"   Created embeddings with shape: {embeddings_array.shape}")
    
    # Save
    print(f"💾 Saving to {OUTPUT_PATH}...")
    np.save(OUTPUT_PATH, embeddings_array)
    
    # Report file size
    import os
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"   Saved! File size: {size_mb:.1f} MB")
    
    print("\n✅ Done! You can now deploy your app.")
    print("   Files to upload:")
    print("   - app.py")
    print("   - requirements.txt")
    print("   - berntsen_dictionary_processed.json")
    print("   - embeddings.npy")

if __name__ == "__main__":
    main()
