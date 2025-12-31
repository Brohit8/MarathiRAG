# 🇮🇳 Marathi-English Dictionary RAG

A smart Marathi-English dictionary that understands meaning, not just exact matches.

## Quick Start (Phase 1)

### 1. Create Project Folder

On your Mac, create this folder structure:

```
marathi-dictionary/
├── notebooks/
│   └── 01_embeddings_and_search.ipynb
├── data/
│   └── berntsen_dictionary_processed.json   ← Copy your JSON here!
├── chroma_db/                                ← Created automatically
└── requirements.txt
```

### 2. Set Up Python Environment

Open Terminal and run:

```bash
# Navigate to your project folder
cd path/to/marathi-dictionary

# Create a virtual environment (keeps packages separate from system Python)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv) at the start

# Install packages
pip install -r requirements.txt
```

### 3. Copy Your Dictionary Data

Copy `berntsen_dictionary_processed.json` to the `data/` folder.

### 4. Open in VS Code

```bash
# Open VS Code in the project folder
code .
```

### 5. Run the Notebook

1. In VS Code, open `notebooks/01_embeddings_and_search.ipynb`
2. VS Code will ask to install the Jupyter extension - click Yes
3. Select your Python interpreter: click the kernel picker (top right) → select the `venv` you created
4. Run cells one by one with Shift+Enter

## What Each Cell Does

| Cell | What It Does | Time |
|------|--------------|------|
| Step 1 | Checks packages are installed | Instant |
| Step 2 | Loads MahaSBERT model | 1-2 min (first time) |
| Step 3 | Shows how embeddings work | Instant |
| Step 4 | Loads your dictionary | Instant |
| Step 5 | Creates embeddings for all entries | 2-5 min |
| Step 6 | Stores in ChromaDB | 1-2 min |
| Step 7 | Search! | Instant |

## Troubleshooting

### "Module not found" error
Make sure you activated the virtual environment:
```bash
source venv/bin/activate
```

### "File not found" error for dictionary
Make sure `berntsen_dictionary_processed.json` is in the `data/` folder.

### VS Code doesn't recognize Jupyter
Install the Jupyter extension: Cmd+Shift+X → search "Jupyter" → Install

### Model download is slow
First-time download is ~400MB. Be patient or use a faster network.

## Next Steps

After completing Phase 1:
- Phase 2: Add LLM (Claude Haiku) for smarter responses
- Phase 3: Build FastAPI backend
- Phase 4: Create chat interface
- Phase 5: Add more dictionaries
- Phase 6: Deploy!
