"""
Download and combine Kafka + Dostoyevsky texts from Project Gutenberg,
then rebuild the word-level tokenizer vocabulary.

Usage:
    python utils/prepare_dataset.py

Outputs:
    kafka_dostoyevsky.txt   — combined cleaned corpus
    tokens.pkl              — word→id vocab (overwrites Shakespeare vocab)
    inv_tokens.pkl          — id→word vocab (overwrites Shakespeare vocab)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import urllib.request
import re
from tokenizer import Tokenizer

WORKS = {
    # Kafka (English translations on Project Gutenberg)
    "Kafka – The Metamorphosis":    5200,
    "Kafka – The Trial":            7849,
    # Dostoyevsky (Constance Garnett translations)
    "Dostoyevsky – Notes from the Underground": 600,
    "Dostoyevsky – The Gambler":    2197,
    "Dostoyevsky – Poor Folk":      2302,
    "Dostoyevsky – White Nights":   36034,
    "Dostoyevsky – Crime and Punishment": 2554,
}

OUTPUT_FILE = "kafka_dostoyevsky.txt"


def download_pg_text(pg_id: int) -> str:
    """Fetch plain UTF-8 text from Project Gutenberg, trying common URL patterns."""
    urls = [
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}-0.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ShakespeareLM dataset builder)"}
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            print(f"    OK  {url}")
            return text
        except Exception:
            continue
    raise RuntimeError(
        f"Could not download PG #{pg_id}. Check your internet connection or "
        f"visit https://www.gutenberg.org/ebooks/{pg_id} to download manually."
    )


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove PG license header and footer, keeping only the literary text."""
    start_pat = re.compile(
        r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
        re.IGNORECASE,
    )
    end_pat = re.compile(
        r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
        re.IGNORECASE,
    )

    m = start_pat.search(text)
    if m:
        text = text[m.end():]

    m = end_pat.search(text)
    if m:
        text = text[: m.start()]

    return text.strip()


def main():
    sections = []
    total_words = 0

    for title, pg_id in WORKS.items():
        print(f"Downloading: {title} (PG #{pg_id})")
        raw = download_pg_text(pg_id)
        cleaned = strip_gutenberg_boilerplate(raw)
        word_count = len(cleaned.split())
        total_words += word_count
        print(f"    {word_count:,} words after stripping boilerplate")
        sections.append(cleaned)

    full_text = "\n\n".join(sections)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\nSaved '{OUTPUT_FILE}'  ({total_words:,} words total)")

    # Rebuild tokenizer vocabulary from the new corpus
    print("\nRebuilding tokenizer vocabulary...")
    unique_tokens = Tokenizer.index_tokens(full_text)
    pad_token_id = unique_tokens          # first ID beyond real vocab
    model_vocab_size = unique_tokens + 1  # embedding table must cover pad token

    print(f"\n{'='*50}")
    print(f"  Corpus word count:    {total_words:,}")
    print(f"  Unique tokens:        {unique_tokens:,}")
    print(f"  Padding token ID:     {pad_token_id}")
    print(f"  Model vocab_size:     {model_vocab_size}")
    print(f"{'='*50}")
    print(
        "\nTokenizer saved to tokens.pkl / inv_tokens.pkl.\n"
        "train.py will pick up the correct vocab_size automatically."
    )


if __name__ == "__main__":
    main()
