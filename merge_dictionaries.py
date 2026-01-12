#!/usr/bin/env python3
"""
Merge Berntsen V2 and Molesworth dictionaries into a single combined dictionary.
Both dictionaries already share the same schema.
"""

import json
from pathlib import Path

def merge_dictionaries():
    """Merge Berntsen V2 and Molesworth dictionaries"""

    print("=" * 70)
    print("MERGING DICTIONARIES")
    print("=" * 70)

    # Load Berntsen V2
    print("\n📖 Loading Berntsen V2...")
    with open('berntsen_dictionary_v2.json', 'r', encoding='utf-8') as f:
        berntsen = json.load(f)
    print(f"   ✅ Loaded {len(berntsen):,} Berntsen entries")

    # Load Molesworth
    print("\n📗 Loading Molesworth...")
    with open('molesworth_dictionary.json', 'r', encoding='utf-8') as f:
        molesworth = json.load(f)
    print(f"   ✅ Loaded {len(molesworth):,} Molesworth entries")

    # Verify schemas match
    print("\n🔍 Verifying schemas...")
    berntsen_keys = set(berntsen[0].keys())
    molesworth_keys = set(molesworth[0].keys())

    if berntsen_keys == molesworth_keys:
        print(f"   ✅ Schemas match perfectly!")
        print(f"   Fields: {sorted(berntsen_keys)}")
    else:
        print(f"   ⚠️  Schema mismatch detected!")
        print(f"   Only in Berntsen: {berntsen_keys - molesworth_keys}")
        print(f"   Only in Molesworth: {molesworth_keys - berntsen_keys}")

    # Merge dictionaries
    print("\n🔀 Merging dictionaries...")
    combined = berntsen + molesworth
    print(f"   ✅ Combined: {len(combined):,} total entries")

    # Statistics
    print("\n📊 Statistics:")
    berntsen_words = set(e['headword_devanagari'] for e in berntsen)
    molesworth_words = set(e['headword_devanagari'] for e in molesworth)
    overlap = berntsen_words & molesworth_words
    unique_words = berntsen_words | molesworth_words

    print(f"   Berntsen entries:       {len(berntsen):,}")
    print(f"   Molesworth entries:     {len(molesworth):,}")
    print(f"   Total entries:          {len(combined):,}")
    print(f"   Unique words:           {len(unique_words):,}")
    print(f"   Words in both:          {len(overlap):,}")
    print(f"   Only in Berntsen:       {len(berntsen_words - molesworth_words):,}")
    print(f"   Only in Molesworth:     {len(molesworth_words - berntsen_words):,}")

    # Save combined dictionary
    output_file = 'combined_dictionary.json'
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # Check file size
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved successfully!")
    print(f"   File size: {file_size_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Generate embeddings: python generate_embeddings.py")
    print(f"2. Update app.py to use 'combined_dictionary.json'")
    print(f"3. Test locally: python app.py")
    print(f"4. Deploy to HF Spaces with Git LFS")

    return combined

if __name__ == "__main__":
    try:
        combined = merge_dictionaries()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure these files exist:")
        print(f"  - berntsen_dictionary_v2.json")
        print(f"  - molesworth_dictionary.json")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
