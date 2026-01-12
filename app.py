"""
Marathi-English Dictionary
A semantic search dictionary powered by MahaSBERT

Deploy to Hugging Face Spaces for free hosting.
"""

import json
import os
import gradio as gr
import numpy as np
from pathlib import Path
from anthropic import Anthropic

# ============================================================
# CONFIGURATION
# ============================================================

DICTIONARY_PATH = "combined_dictionary.json"
EMBEDDINGS_PATH = "combined_embeddings.npy"
MODEL_NAME = "l3cube-pune/marathi-sentence-similarity-sbert"
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score (60%) to show results
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Get API key from environment

# Initialize Anthropic client (will be None if API key not set)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Response cache for repeated queries
response_cache = {}

# Morphology cache
morphology_cache = {}

# ============================================================
# MARATHI MORPHOLOGY RULES
# ============================================================

# Based on Marathi Grammar (Dhongde & Wali, 2009) and Pandharipande (1997)
MARATHI_MORPHOLOGY = {
    # Case suffixes (vibhakti) - most common postpositions
    'case_suffixes': {
        'ला': {'type': 'dative', 'meaning': 'to/for', 'sandhi': 'ī→yā'},
        'ने': {'type': 'instrumental', 'meaning': 'by/with', 'sandhi': 'ī→ye'},
        'त': {'type': 'locative', 'meaning': 'in/at', 'sandhi': None},
        'स': {'type': 'locative', 'meaning': 'to', 'sandhi': None},
        'वर': {'type': 'locative', 'meaning': 'on', 'sandhi': None},
        'मध्ये': {'type': 'locative', 'meaning': 'in/within', 'sandhi': None},
        'शी': {'type': 'comitative', 'meaning': 'with', 'sandhi': None},
        'पासून': {'type': 'ablative', 'meaning': 'from', 'sandhi': None},
        'साठी': {'type': 'benefactive', 'meaning': 'for', 'sandhi': None},
        'कडे': {'type': 'locative', 'meaning': 'near/to', 'sandhi': None},
    },

    # Possessive markers (sambandh vibhakti)
    'possessive': {
        'चा': {'gender': 'masculine', 'meaning': 'of'},
        'ची': {'gender': 'feminine', 'meaning': 'of'},
        'चे': {'gender': 'neuter', 'meaning': 'of'},
        'च्या': {'case': 'oblique', 'meaning': 'of'},
    },
}

def reverse_sandhi(word: str, sandhi_type: str) -> str:
    """
    Reverse common sandhi changes to get dictionary form

    Marathi sandhi rules:
    - ी (ī) → ्या (yā) before ला: पाणी → पाण्याला
    - ी (ī) → े (e) before ने: पाणी → पाण्याने
    """
    if not sandhi_type:
        return word

    # Handle ī→yā sandhi (most common)
    # पाण्या → पाणी (remove ्या, add ी)
    if sandhi_type == 'ī→yā':
        # पाण्या is: प + ा + ण + ् + य + ा
        # We want: प + ा + ण + ी
        if word.endswith('्या'):
            # Remove last 3 chars (्या) and add ी
            return word[:-3] + 'ी'
        elif word.endswith('ा'):
            # Simple case: just replace final आ with ई
            return word[:-1] + 'ी'

    # Handle ī→ye sandhi
    # पाणे → पाणी
    if sandhi_type == 'ī→ye':
        if word.endswith('े'):
            return word[:-1] + 'ी'

    return word

def detect_marathi_morphology(query: str) -> dict:
    """
    Fast rule-based morphology detection for Marathi

    Returns:
        dict with keys:
        - has_morphology: bool
        - root: str (stem after removing suffix)
        - original_root: str (dictionary form after sandhi reversal)
        - suffix: str
        - morphology_type: str
        - explanation: str
        - search_variants: list[str] (terms to search)
    """
    # Check cache first
    if query in morphology_cache:
        return morphology_cache[query]

    query_normalized = query.strip()

    # Check case suffixes (longer suffixes first to avoid partial matches)
    sorted_suffixes = sorted(
        MARATHI_MORPHOLOGY['case_suffixes'].items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for suffix, info in sorted_suffixes:
        if query_normalized.endswith(suffix):
            stem = query_normalized[:-len(suffix)]

            # Apply sandhi reversal to get dictionary form
            original_root = reverse_sandhi(stem, info.get('sandhi'))

            result = {
                'has_morphology': True,
                'root': stem,
                'original_root': original_root,
                'suffix': suffix,
                'morphology_type': info['type'],
                'explanation': f"{info['meaning']} ({info['type']} case)",
                'search_variants': [original_root, stem, query_normalized]
            }

            morphology_cache[query] = result
            return result

    # Check possessive markers
    for suffix, info in MARATHI_MORPHOLOGY['possessive'].items():
        if query_normalized.endswith(suffix):
            stem = query_normalized[:-len(suffix)]

            result = {
                'has_morphology': True,
                'root': stem,
                'original_root': stem,
                'suffix': suffix,
                'morphology_type': 'possessive',
                'explanation': f"{info['meaning']} (possessive {info.get('gender', 'oblique')})",
                'search_variants': [stem, query_normalized]
            }

            morphology_cache[query] = result
            return result

    # No morphology detected
    result = {
        'has_morphology': False,
        'search_variants': [query_normalized]
    }

    morphology_cache[query] = result
    return result

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("🔄 Loading dictionary...")
with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
    dictionary = json.load(f)
print(f"✅ Loaded {len(dictionary):,} entries")

# Build lookup index for instant exact matches (O(1) instead of O(n))
print("🔄 Building search index...")
headword_index = {}  # Maps lowercase headword -> list of indices
for idx, entry in enumerate(dictionary):
    # Index by Devanagari headword
    key = entry['headword_devanagari'].lower()
    if key not in headword_index:
        headword_index[key] = []
    headword_index[key].append(idx)

    # Index by romanized headword if available
    if entry.get('headword_romanized'):
        key_roman = entry['headword_romanized'].lower()
        if key_roman not in headword_index:
            headword_index[key_roman] = []
        headword_index[key_roman].append(idx)
print(f"✅ Indexed {len(headword_index):,} unique headwords")

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
    """Hybrid search with exact match priority, similarity threshold, and optional entry type filter"""
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

    # Add exact matches first (always show, regardless of threshold)
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

    # Add semantic search results (excluding exact matches, applying threshold)
    for idx in sorted_indices:
        if count >= num_results:
            break

        # Skip if already added as exact match
        if idx in seen_indices:
            continue

        # Apply similarity threshold (only for semantic matches, not exact)
        score = similarities[idx]
        if score < SIMILARITY_THRESHOLD:
            continue

        entry = dictionary[idx]

        # Apply filter
        if entry_type != "All" and entry['entry_type'] != entry_type.lower():
            continue

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

    # Return appropriate message based on results
    if not results:
        return f"No good matches found for '{query}'. Try a different spelling or related word."

    # Add header showing how many results passed the threshold
    threshold_pct = int(SIMILARITY_THRESHOLD * 100)
    if count > 0:
        header_msg = f"**Showing {count} result{'s' if count != 1 else ''} above {threshold_pct}% match**\n\n---\n\n"
        return header_msg + "\n".join(results)

    return "\n".join(results)

# ============================================================
# LLM INTEGRATION
# ============================================================

def generate_llm_response(query: str, search_results_data: list, morph_info: dict = None) -> str:
    """
    Generate an intelligent response using Claude Haiku based on search results.

    Args:
        query: The user's search query
        search_results_data: List of dict entries from the dictionary
        morph_info: Optional morphology detection results

    Returns:
        LLM-generated response as markdown string
    """
    if not anthropic_client:
        return ""  # Silently fail if no API key

    # Check cache first
    morph_key = ""
    if morph_info and morph_info.get('has_morphology'):
        morph_key = f":{morph_info['suffix']}"
    cache_key = f"{query}:{len(search_results_data)}{morph_key}"
    if cache_key in response_cache:
        return response_cache[cache_key]

    # Prepare search results summary for the LLM
    results_summary = []
    for i, entry in enumerate(search_results_data, 1):
        headword = entry['headword_devanagari']
        romanized = entry.get('headword_romanized', '')
        full_entry = entry['full_entry']
        page = entry['source_page']
        score = entry.get('similarity_score', 0)
        is_exact = entry.get('is_exact_match', False)

        result_text = f"""Result {i}{"(EXACT MATCH)" if is_exact else ""}:
Headword: {headword}
Romanized: {romanized}
Full Entry: {full_entry}
Page: {page}
Similarity: {score:.1%}
"""
        results_summary.append(result_text)

    results_text = "\n".join(results_summary)

    # System prompt for Claude Haiku
    system_prompt = """You are a Marathi-English dictionary assistant. Provide simple, structured responses.

OUTPUT FORMAT (follow exactly):

For exact matches:
पाणी (pāṇī) n. water

━━━━━━━━━━━━━━━━━━━━━
Source: Berntsen p.542

For inflected forms:
पाण्याला is the dative case of पाणी (pāṇī)

पाणी (pāṇī) n. water
→ -ला (-lā) = dative suffix "to/for"

━━━━━━━━━━━━━━━━━━━━━
Source: Berntsen p.542

For multiple meanings:
घर (ghar) n. 1. house, home 2. household

━━━━━━━━━━━━━━━━━━━━━
Source: Berntsen p.234

RULES:
1. First line: Headword (romanization) part-of-speech. definition
2. For inflected forms: explain root + suffix
3. Keep definitions brief (under 10 words)
4. Always end with source citation using page number from search results
5. Use Devanagari for Marathi words
6. Use → for morphology explanations
7. Common suffixes: ला (lā-dative), ने (ne-instrumental), चा/ची/चे (possessive), स/त (locative)"""

    # Build morphology context if detected
    morph_context = ""
    if morph_info and morph_info.get('has_morphology'):
        morph_context = f"""
Morphology Analysis (pre-computed):
- Original query: {query}
- Root word: {morph_info['original_root']}
- Detected suffix: {morph_info['suffix']}
- Type: {morph_info['morphology_type']}
- Meaning: {morph_info['explanation']}

Note: The morphology has been pre-analyzed. Confirm this analysis and present in standard format.
"""

    user_prompt = f"""Query: "{query}"
{morph_context}
Search results:
{results_text}

Provide a structured response following the format exactly. Use the page number from the best matching result."""

    try:
        # Call Claude Haiku
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.3,  # Lower temperature for more consistent responses
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        response_text = message.content[0].text

        # Cache the response
        response_cache[cache_key] = response_text

        return response_text

    except Exception as e:
        # Silently fail - raw results will still be shown
        print(f"LLM error: {e}")
        return ""

def get_search_results_data(query: str, num_results: int, entry_type: str) -> tuple:
    """
    Perform search and return both structured data and formatted results.

    Returns:
        (results_data, formatted_results_string, morphology_info)
    """
    if not query.strip():
        return [], "Please enter a word to search."

    # Step 1: Detect morphology
    morph = detect_marathi_morphology(query)

    # Get search variants (includes original query + morphological roots)
    search_variants = morph['search_variants']

    # Step 2: Find exact matches using index (O(1) lookup)
    exact_match_indices = []
    seen = set()
    for search_term in search_variants:
        query_normalized = search_term.strip().lower()
        # Use index for instant lookup instead of looping through 89k entries
        for idx in headword_index.get(query_normalized, []):
            if idx not in seen:
                exact_match_indices.append(idx)
                seen.add(idx)

    # Step 3: Only run semantic search if we need more results
    # Skip expensive embedding computation if we have enough exact matches
    if len(exact_match_indices) >= num_results:
        # We have enough exact matches, no need for semantic search
        similarities = np.zeros(len(dictionary))
        for idx in exact_match_indices:
            similarities[idx] = 1.0  # Mark exact matches with perfect score
        sorted_indices = []  # No semantic results needed
    else:
        # Not enough exact matches, fall back to semantic search
        query_embedding = model.encode(query)
        similarities = cosine_similarity(query_embedding, embeddings)
        sorted_indices = np.argsort(similarities)[::-1]

    # Step 3: Build results
    results_data = []
    results_formatted = []
    count = 0
    seen_indices = set()

    # Add exact matches first
    for idx in exact_match_indices:
        entry = dictionary[idx]

        if entry_type != "All" and entry['entry_type'] != entry_type.lower():
            continue

        score = similarities[idx]
        count += 1
        seen_indices.add(idx)

        # Store structured data
        entry_data = entry.copy()
        entry_data['similarity_score'] = score
        entry_data['is_exact_match'] = True
        results_data.append(entry_data)

        # Format for display
        headword = entry['headword_devanagari']
        romanized = entry.get('headword_romanized', '')
        full_entry = entry['full_entry']
        etype = entry['entry_type']
        page = entry['source_page']

        if romanized:
            header = f"### {count}. {headword} ({romanized}) ⭐ Exact match"
        else:
            header = f"### {count}. {headword} ⭐ Exact match"

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

        card = f"""{header}

{full_entry}

{def_text}
📖 *{etype}* · Page {page} · Match: {score:.1%}

---
"""
        results_formatted.append(card)

    # Add semantic search results
    for idx in sorted_indices:
        if count >= num_results:
            break

        if idx in seen_indices:
            continue

        score = similarities[idx]
        if score < SIMILARITY_THRESHOLD:
            continue

        entry = dictionary[idx]

        if entry_type != "All" and entry['entry_type'] != entry_type.lower():
            continue

        count += 1
        seen_indices.add(idx)

        # Store structured data
        entry_data = entry.copy()
        entry_data['similarity_score'] = score
        entry_data['is_exact_match'] = False
        results_data.append(entry_data)

        # Format for display
        headword = entry['headword_devanagari']
        romanized = entry.get('headword_romanized', '')
        full_entry = entry['full_entry']
        etype = entry['entry_type']
        page = entry['source_page']

        if romanized:
            header = f"### {count}. {headword} ({romanized})"
        else:
            header = f"### {count}. {headword}"

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

        card = f"""{header}

{full_entry}

{def_text}
📖 *{etype}* · Page {page} · Match: {score:.1%}

---
"""
        results_formatted.append(card)

    # Format final output
    if not results_formatted:
        return [], f"No good matches found for '{query}'. Try a different spelling or related word.", morph

    threshold_pct = int(SIMILARITY_THRESHOLD * 100)
    if count > 0:
        header_msg = f"**Showing {count} result{'s' if count != 1 else ''} above {threshold_pct}% match**\n\n---\n\n"
        formatted_output = header_msg + "\n".join(results_formatted)
    else:
        formatted_output = "\n".join(results_formatted)

    return results_data, formatted_output, morph

def search_with_llm(query: str, num_results: int, entry_type: str) -> tuple:
    """
    Search with LLM-enhanced response.

    Returns:
        (llm_response, raw_results)
    """
    # Get search results with morphology info
    results_data, formatted_results, morph_info = get_search_results_data(query, min(5, num_results), entry_type)

    # Generate LLM response if available (pass morphology info)
    llm_response = ""
    if results_data and anthropic_client:
        llm_response = generate_llm_response(query, results_data, morph_info)

    # Get full results for display
    _, full_formatted_results, _ = get_search_results_data(query, num_results, entry_type)

    return llm_response, full_formatted_results

# ============================================================
# GRADIO INTERFACE
# ============================================================

# Custom CSS for better appearance (theme-aware)
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.result-box {
    font-family: 'Noto Sans Devanagari', sans-serif;
}
/* AI Assistant box - adapts to light/dark theme */
.ai-assistant-box {
    background-color: var(--block-background-fill);
    border-left: 4px solid var(--color-accent);
    padding: 15px;
    border-radius: var(--radius-lg);
    font-family: 'Noto Sans Devanagari', sans-serif;
    margin-bottom: 20px;
    border: 1px solid var(--border-color-primary);
}
.ai-assistant-box p {
    color: var(--body-text-color) !important;
}
.ai-assistant-box code {
    background-color: var(--background-fill-secondary);
    color: var(--body-text-color);
    padding: 2px 4px;
    border-radius: 3px;
}
"""

# Build the interface
with gr.Blocks(title="Marathi Dictionary") as app:
    gr.Markdown("""
    # 🇮🇳 Marathi-English Dictionary

    **Semantic search** powered by MahaSBERT + Claude Haiku AI assistant. Search in Marathi or English -
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

    # LLM Response section (only shows if API key is set)
    if anthropic_client:
        llm_output = gr.Markdown(
            label="📖 Dictionary Definition",
            elem_classes=["ai-assistant-box"],
            visible=True
        )

    # Raw results section
    results_output = gr.Markdown(
        label="Dictionary Results",
        elem_classes=["result-box"]
    )

    # Example searches
    gr.Markdown("### Try these examples:")
    if anthropic_client:
        gr.Examples(
            examples=[
                ["पाणी", 10, "All"],
                ["water", 10, "All"],
                ["आई", 10, "All"],
                ["पाण्याला", 10, "All"],  # Inflected form example
                ["खाणे", 10, "Collocation"],
                ["beautiful", 10, "All"],
                ["घर", 10, "All"],
            ],
            inputs=[query_input, num_results, entry_type_filter],
            outputs=[llm_output, results_output],
            fn=search_with_llm,
        )
    else:
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
    if anthropic_client:
        search_btn.click(
            fn=search_with_llm,
            inputs=[query_input, num_results, entry_type_filter],
            outputs=[llm_output, results_output]
        )

        query_input.submit(
            fn=search_with_llm,
            inputs=[query_input, num_results, entry_type_filter],
            outputs=[llm_output, results_output]
        )
    else:
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
    *Source: Berntsen Marathi-English Dictionary · Embeddings: MahaSBERT (L3Cube Pune) · AI: Claude Haiku (Anthropic)*
    """)

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    app.launch(css=css)
