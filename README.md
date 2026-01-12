---
title: Marathirag
emoji: 🌍
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
short_description: AI-powered Marathi-English dictionary with semantic search
---

# 🇮🇳 Marathi-English Dictionary

An intelligent dictionary application powered by semantic search and AI assistance.

## Features

- **Hybrid Search**: Combines exact string matching with semantic vector search
- **AI Assistant**: Claude Haiku LLM analyzes results and explains morphology, inflected forms, and provides context
- **Semantic Understanding**: Powered by MahaSBERT (L3Cube Pune) - understands meaning, not just exact matches
- **Morphology Detection**: Recognizes inflected forms (e.g., "पाण्याला" → "पाणी" + dative suffix)
- **Similarity Threshold**: Only shows results above 60% match quality
- **Bilingual**: Search in Marathi (Devanagari), English, or romanized Marathi

## Data Source

Based on the **Berntsen Marathi-English Dictionary** with 50,000+ entries including headwords and collocations.

## Setup

### Local Development

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Claude Haiku API key for AI features:
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

4. Run the app:
```bash
python app.py
```

### Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude Haiku integration (optional)
  - Get your key from: https://console.anthropic.com/
  - The app works without this, but won't have AI-powered explanations

## How It Works

1. **User Query**: Enter a word in Marathi or English
2. **Hybrid Search**:
   - Exact match check on headword fields
   - Semantic similarity using MahaSBERT embeddings
   - Results ranked by relevance
3. **AI Analysis** (if API key set):
   - Claude Haiku receives top 5 results
   - Analyzes morphology and context
   - Provides clear, concise explanation
4. **Results Display**:
   - AI explanation (if available)
   - Full dictionary entries with definitions
   - Page references and match scores

## Technical Stack

- **Frontend**: Gradio 6.2
- **Embeddings**: sentence-transformers (MahaSBERT)
- **Vector Search**: NumPy cosine similarity
- **AI**: Claude Haiku 3 (Anthropic)
- **Dictionary**: Berntsen Marathi-English Dictionary (processed JSON)

## Examples

- `पाणी` - Find "water" in Marathi
- `पाण्याला` - AI explains this is dative form of "पाणी"
- `water` - Search English word for Marathi translation
- `खाणे` - Find collocations with "to eat"

## Deployment to Hugging Face Spaces

1. **Upload files** via web UI or Git:
   - `app.py`, `requirements.txt`, `README.md`
   - `berntsen_dictionary_processed.json` (dictionary data)
   - `embeddings.npy` (precomputed embeddings - use Git LFS for large files)

2. **Add API key** in Space Settings → Repository secrets:
   - Name: `ANTHROPIC_API_KEY`
   - Value: Your API key from https://console.anthropic.com/

3. **Done!** Space will auto-deploy and restart.

See [docs/](docs/) for detailed technical documentation.

## License

Dictionary data from Berntsen Marathi-English Dictionary.
