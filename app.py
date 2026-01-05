"""
Marathi-English Dictionary
A semantic search dictionary powered by MahaSBERT

Deploy to Hugging Face Spaces for free hosting.
"""

import json
import gradio as gr
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DICTIONARY_PATH = "berntsen_dictionary_processed.json"
EMBEDDINGS_PATH = "embeddings.npy"
MODEL_NAME = "l3cube-pune/marathi-sentence-similarity-sbert"

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("🔄 Loading dictionary...")
with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
    dictionary = json.load(f)
print(f"✅ Loaded {len(dictionary):,} entries")

print("🔄 Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
print(f"✅ Loaded embeddings with shape {embeddings.shape}")

print("🔄 Loading MahaSBERT model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded!")

# ============================================================
# SEARCH FUNCTION
# ============================================================

def cosine_similarity(a, b):
    """Calculate cosine similarity between vector a and matrix b"""
    # Normalize
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    # Dot product
    return np.dot(b_norm, a_norm)

def search(query: str, num_results: int = 10) -> str:
    """
    Search the dictionary for entries matching the query.
    Returns formatted results as a string.
    """
    if not query.strip():
        return "Please enter a word to search."
    
    # Encode query
    query_embedding = model.encode(query)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:num_results]
    
    # Format results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        entry = dictionary[idx]
        score = similarities[idx]
        
        # Build result card
        headword = entry['headword_devanagari']
        romanized = entry.get('headword_romanized', '')
        full_entry = entry['full_entry']
        entry_type = entry['entry_type']
        page = entry['source_page']
        
        # Format header
        if romanized:
            header = f"### {rank}. {headword} ({romanized})"
        else:
            header = f"### {rank}. {headword}"
        
        # Format definitions
        definitions = entry.get('definitions', [])
        def_text = ""
        if definitions:
            for d in definitions:
                pos = d.get('pos_display', '')
                defn = d.get('definition', '')
                num = d.get('number', '')
                if num:
                    def_text += f"  {num}. {defn} *{pos}*\n"
                else:
                    def_text += f"  • {defn} *{pos}*\n"
        
        # Build card
        card = f"""{header}

{full_entry}

{def_text}
📖 *{entry_type}* · Page {page} · Match: {score:.1%}

---
"""
        results.append(card)
    
    return "\n".join(results)


def search_with_filters(query: str, num_results: int, entry_type: str) -> str:
    """Hybrid search with exact match priority and optional entry type filter"""
    if not query.strip():
        return "Please enter a word to search."

    query_normalized = query.strip().lower()

    # Step 1: Find exact matches
    exact_match_indices = []
    for idx, entry in enumerate(dictionary):
        # Check headword_devanagari
        if entry['headword_devanagari'].lower() == query_normalized:
            exact_match_indices.append(idx)
        # Check headword_romanized
        elif entry.get('headword_romanized') and entry['headword_romanized'].lower() == query_normalized:
            exact_match_indices.append(idx)

    # Step 2: Encode query and calculate semantic similarities
    query_embedding = model.encode(query)
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get sorted indices by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # Step 3: Build results with exact matches first
    results = []
    count = 0
    seen_indices = set()

    # Add exact matches first
    for idx in exact_match_indices:
        entry = dictionary[idx]

        # Apply filter
        if entry_type != "All" and entry['entry_type'] != entry_type.lower():
            continue

        score = similarities[idx]
        count += 1
        seen_indices.add(idx)

        # Build result card with exact match indicator
        headword = entry['headword_devanagari']
        romanized = entry.get('headword_romanized', '')
        full_entry = entry['full_entry']
        etype = entry['entry_type']
        page = entry['source_page']

        # Format header with exact match indicator
        if romanized:
            header = f"### {count}. {headword} ({romanized}) ⭐ Exact match"
        else:
            header = f"### {count}. {headword} ⭐ Exact match"

        # Format definitions
        definitions = entry.get('definitions', [])
        def_text = ""
        if definitions:
            for d in definitions:
                pos = d.get('pos_display', '')
                defn = d.get('definition', '')
                num = d.get('number', '')
                if num:
                    def_text += f"  {num}. {defn} *{pos}*\n"
                else:
                    def_text += f"  • {defn} *{pos}*\n"

        # Build card
        card = f"""{header}

{full_entry}

{def_text}
📖 *{etype}* · Page {page} · Match: {score:.1%}

---
"""
        results.append(card)

    # Add semantic search results (excluding exact matches)
    for idx in sorted_indices:
        if count >= num_results:
            break

        # Skip if already added as exact match
        if idx in seen_indices:
            continue

        entry = dictionary[idx]

        # Apply filter
        if entry_type != "All" and entry['entry_type'] != entry_type.lower():
            continue

        score = similarities[idx]
        count += 1
        seen_indices.add(idx)

        # Build result card
        headword = entry['headword_devanagari']
        romanized = entry.get('headword_romanized', '')
        full_entry = entry['full_entry']
        etype = entry['entry_type']
        page = entry['source_page']

        # Format header
        if romanized:
            header = f"### {count}. {headword} ({romanized})"
        else:
            header = f"### {count}. {headword}"

        # Format definitions
        definitions = entry.get('definitions', [])
        def_text = ""
        if definitions:
            for d in definitions:
                pos = d.get('pos_display', '')
                defn = d.get('definition', '')
                num = d.get('number', '')
                if num:
                    def_text += f"  {num}. {defn} *{pos}*\n"
                else:
                    def_text += f"  • {defn} *{pos}*\n"

        # Build card
        card = f"""{header}

{full_entry}

{def_text}
📖 *{etype}* · Page {page} · Match: {score:.1%}

---
"""
        results.append(card)

    if not results:
        return "No results found. Try a different search term."

    return "\n".join(results)

# ============================================================
# GRADIO INTERFACE
# ============================================================

# Custom CSS for better appearance
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.result-box {
    font-family: 'Noto Sans Devanagari', sans-serif;
}
"""

# Build the interface
with gr.Blocks(title="Marathi Dictionary") as app:
    gr.Markdown("""
    # 🇮🇳 Marathi-English Dictionary
    
    **Semantic search** powered by MahaSBERT. Search in Marathi or English - 
    the dictionary understands meaning, not just exact matches.
    
    *Based on the Berntsen Marathi-English Dictionary*
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Search",
                placeholder="Enter a Marathi word (पाणी) or English word (water)...",
                lines=1
            )
        with gr.Column(scale=1):
            num_results = gr.Slider(
                minimum=1,
                maximum=25,
                value=10,
                step=1,
                label="Results"
            )
    
    with gr.Row():
        entry_type_filter = gr.Radio(
            choices=["All", "Headword", "Collocation"],
            value="All",
            label="Entry Type"
        )
        search_btn = gr.Button("🔍 Search", variant="primary")
    
    results_output = gr.Markdown(
        label="Results",
        elem_classes=["result-box"]
    )
    
    # Example searches
    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=[
            ["पाणी", 10, "All"],
            ["water", 10, "All"],
            ["आई", 10, "All"],
            ["खाणे", 10, "Collocation"],
            ["beautiful", 10, "All"],
            ["घर", 10, "All"],
        ],
        inputs=[query_input, num_results, entry_type_filter],
        outputs=results_output,
        fn=search_with_filters,
    )
    
    # Wire up the search
    search_btn.click(
        fn=search_with_filters,
        inputs=[query_input, num_results, entry_type_filter],
        outputs=results_output
    )
    
    query_input.submit(
        fn=search_with_filters,
        inputs=[query_input, num_results, entry_type_filter],
        outputs=results_output
    )
    
    gr.Markdown("""
    ---
    *Source: Berntsen Marathi-English Dictionary · Embeddings: MahaSBERT (L3Cube Pune)*
    """)

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    app.launch(css=css)
