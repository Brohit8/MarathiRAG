#!/usr/bin/env python3
"""
DSAL Berntsen Marathi-English Dictionary Scraper

Scrapes all pages of the dictionary from:
https://dsal.uchicago.edu/cgi-bin/app/berntsen_query.py?page=N

Each entry is contained in a <div> within <div class='hw_result'>.
Structure:
    <div>
        <hw><d><b>Devanagari</b> <b>romanized</b></d></hw> <i>POS</i> definition...
    </div>

Usage:
    python berntsen_scraper.py                    # Scrape all pages
    python berntsen_scraper.py --start 1 --end 10 # Scrape pages 1-10
    python berntsen_scraper.py --page 30          # Scrape single page
"""

import argparse
import json
import re
import sys
import time

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install requests beautifulsoup4 lxml")
    sys.exit(1)


# Constants
BASE_URL = "https://dsal.uchicago.edu/cgi-bin/app/berntsen_query.py?page={}"
DEFAULT_DELAY = 0.3

# Regex patterns
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]+')
POS_PATTERN = re.compile(
    r'\b(m\.|f\.|n\.|adj\.|adv\.|v\.t\.|v\.i\.|interj\.|post\.|suff\.|abbrev\.)'
    r'(\s*inv\.)?'
    r'(\s*\([^)]*\))?',
    re.IGNORECASE
)


def fetch_page(page_num: int, session: requests.Session) -> str:
    """Fetch a single page from the dictionary."""
    url = BASE_URL.format(page_num)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.RequestException as e:
        print(f"  Error fetching page {page_num}: {e}")
        return ""


def detect_last_page(session: requests.Session, max_page: int = 200) -> int:
    """Detect the last valid page by binary search."""
    print("Detecting total number of pages...")
    
    low, high = 1, max_page
    last_valid = 1
    
    while low <= high:
        mid = (low + high) // 2
        html = fetch_page(mid, session)
        
        if html:
            soup = BeautifulSoup(html, 'lxml')
            # Check if page has dictionary entries
            hw_result = soup.find('div', class_='hw_result')
            if hw_result and hw_result.find_all('div'):
                last_valid = mid
                low = mid + 1
            else:
                high = mid - 1
        else:
            high = mid - 1
        
        time.sleep(0.1)
    
    print(f"Detected {last_valid} pages.")
    return last_valid


def extract_entries_from_html(html: str, page_num: int) -> list:
    """
    Extract dictionary entries from HTML.
    
    Each entry is in a <div> inside <div class='hw_result'>.
    Structure:
        <div>
            <hw><d><b>Devanagari</b> <b>romanized</b></d></hw> <i>POS</i> definition...
        </div>
    """
    entries = []
    soup = BeautifulSoup(html, 'lxml')
    
    # Find the results container
    hw_result = soup.find('div', class_='hw_result')
    if not hw_result:
        return entries
    
    # Find all entry divs (direct children that contain actual entries)
    entry_divs = hw_result.find_all('div', recursive=False)
    
    for div in entry_divs:
        # Skip empty divs
        text_content = div.get_text(strip=True)
        if not text_content:
            continue
        
        # Extract headword info from <hw> tag
        hw_tag = div.find('hw')
        if not hw_tag:
            continue
        
        # Get all bold elements within hw tag for headword(s)
        bold_elements = hw_tag.find_all('b')
        if not bold_elements:
            continue
        
        # First bold with Devanagari is the main headword
        headword_dev = ""
        headword_rom = ""
        
        for i, b in enumerate(bold_elements):
            b_text = b.get_text(strip=True)
            if DEVANAGARI_PATTERN.match(b_text):
                if not headword_dev:
                    headword_dev = b_text
                    # Next bold (if exists and not Devanagari) is romanization
                    if i + 1 < len(bold_elements):
                        next_b = bold_elements[i + 1].get_text(strip=True)
                        if not DEVANAGARI_PATTERN.match(next_b):
                            headword_rom = next_b
                    break
        
        if not headword_dev:
            continue
        
        # Get the full entry text (entire div content)
        full_entry = div.get_text(separator=' ', strip=True)
        # Clean up whitespace
        full_entry = ' '.join(full_entry.split())
        
        # Extract part of speech from italic tags or text
        part_of_speech = ""
        # First try to find POS in <i> tags
        italic_tags = div.find_all('i')
        for i_tag in italic_tags:
            i_text = i_tag.get_text(strip=True)
            if POS_PATTERN.match(i_text):
                part_of_speech = i_text
                break
        
        # If not found in italics, try to extract from full text
        if not part_of_speech:
            pos_match = POS_PATTERN.search(full_entry)
            if pos_match:
                part_of_speech = pos_match.group(0).strip()
        
        entry = {
            "headword_devanagari": headword_dev,
            "headword_romanized": headword_rom,
            "part_of_speech": part_of_speech,
            "full_entry": full_entry,
            "source_page": page_num
        }
        entries.append(entry)
    
    return entries


def scrape_dictionary(start_page: int = 1, end_page: int = None,
                      delay: float = DEFAULT_DELAY, verbose: bool = True) -> list:
    """Scrape multiple pages of the dictionary."""
    all_entries = []
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })
    
    # Auto-detect last page if not specified
    if end_page is None:
        end_page = detect_last_page(session)
        if end_page == 0:
            print("Error: Could not detect dictionary pages.")
            return []
    
    if verbose:
        print(f"Scraping pages {start_page} to {end_page}...")
        print(f"Delay between requests: {delay}s")
        print("-" * 50)
    
    for page_num in range(start_page, end_page + 1):
        if verbose:
            print(f"Fetching page {page_num}/{end_page}...", end=" ", flush=True)
        
        html = fetch_page(page_num, session)
        
        if html:
            entries = extract_entries_from_html(html, page_num)
            all_entries.extend(entries)
            
            if verbose:
                print(f"Found {len(entries)} entries")
        else:
            if verbose:
                print("Failed!")
        
        if page_num < end_page:
            time.sleep(delay)
    
    if verbose:
        print("-" * 50)
        print(f"Total entries scraped: {len(all_entries)}")
    
    return all_entries


def save_entries(entries: list, output_path: str, pretty: bool = True):
    """Save entries to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        else:
            json.dump(entries, f, ensure_ascii=False)
    print(f"Saved {len(entries)} entries to {output_path}")


def save_entries_jsonl(entries: list, output_path: str):
    """Save entries to a JSON Lines file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape the DSAL Berntsen Marathi-English Dictionary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python berntsen_scraper.py                      # Scrape all pages
  python berntsen_scraper.py --start 1 --end 10   # Scrape pages 1-10
  python berntsen_scraper.py --page 30            # Scrape only page 30
  python berntsen_scraper.py -o mydict.json       # Custom output file
  python berntsen_scraper.py --jsonl              # Output as JSON Lines

Output Format:
  Each entry contains:
  - headword_devanagari: The headword in Devanagari script
  - headword_romanized: IAST romanization  
  - part_of_speech: Grammatical category (m., f., adj., v.t., etc.)
  - full_entry: Complete entry text from the div
  - source_page: Page number in the dictionary
        """
    )
    
    parser.add_argument('--start', '-s', type=int, default=1,
                        help='First page to scrape (default: 1)')
    parser.add_argument('--end', '-e', type=int, default=None,
                        help='Last page to scrape (default: auto-detect)')
    parser.add_argument('--page', '-p', type=int,
                        help='Scrape a single page only')
    parser.add_argument('--output', '-o', type=str, default='berntsen_dictionary.json',
                        help='Output filename (default: berntsen_dictionary.json)')
    parser.add_argument('--delay', '-d', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between requests in seconds (default: {DEFAULT_DELAY})')
    parser.add_argument('--jsonl', action='store_true',
                        help='Output as JSON Lines format')
    parser.add_argument('--compact', action='store_true',
                        help='Output compact JSON (no indentation)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Handle single page option
    if args.page:
        start_page = args.page
        end_page = args.page
    else:
        start_page = args.start
        end_page = args.end
    
    # Validate
    if start_page < 1:
        print("Error: Start page must be >= 1")
        sys.exit(1)
    if end_page is not None and start_page > end_page:
        print("Error: Start page must be <= end page")
        sys.exit(1)
    
    # Scrape
    entries = scrape_dictionary(
        start_page=start_page,
        end_page=end_page,
        delay=args.delay,
        verbose=not args.quiet
    )
    
    # Save
    if entries:
        if args.jsonl:
            output_path = args.output.replace('.json', '.jsonl') if args.output.endswith('.json') else args.output
            save_entries_jsonl(entries, output_path)
        else:
            save_entries(entries, args.output, pretty=not args.compact)
    else:
        print("No entries were scraped. Check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
