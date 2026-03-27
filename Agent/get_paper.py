#!/usr/bin/env python3
"""
get_paper.py — Fetch recent papers from major OR journals.

Uses the OpenAlex API (free, no key required) to collect metadata and
download open-access PDFs from the past year.

Usage:
  python get_paper.py                          # Last 12 months, all journals
  python get_paper.py --months 6               # Last 6 months
  python get_paper.py --journals "OR,MS"       # Only specific journals
  python get_paper.py --download               # Also download open-access PDFs
  python get_paper.py --max-per-journal 50     # Limit per journal
  python get_paper.py --output-dir Paper/collected

Output:
  <output_dir>/
    papers_metadata.json          # All paper metadata
    papers_summary.csv            # One-line-per-paper summary
    pdfs/                         # Downloaded PDFs (with --download)
      <journal>_<year>_<first_author>.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
#  OR Journal Registry (name -> OpenAlex source ID / ISSN)
#
#  OpenAlex source IDs can be found at: https://api.openalex.org/sources?search=<name>
#  Using ISSNs as fallback for filtering.
# ---------------------------------------------------------------------------

OR_JOURNALS = {
    "OR": {
        "full_name": "Operations Research",
        "issn": ["0030-364X", "1526-5463"],
        "openalex_id": "S125775545",
    },
    "MS": {
        "full_name": "Management Science",
        "issn": ["0025-1909", "1526-5501"],
        "openalex_id": "S185196701",
    },
    "MOR": {
        "full_name": "Mathematics of Operations Research",
        "issn": ["0364-765X", "1526-5471"],
        "openalex_id": "S55826652",
    },
    "IJOC": {
        "full_name": "INFORMS Journal on Computing",
        "issn": ["1091-9856", "1526-5528"],
        "openalex_id": "S49861078",
    },
    "TS": {
        "full_name": "Transportation Science",
        "issn": ["0041-1655", "1526-5447"],
        "openalex_id": "S23280635",
    },
    "MSOM": {
        "full_name": "Manufacturing & Service Operations Management",
        "issn": ["1523-4614", "1526-5498"],
        "openalex_id": "S101820957",
    },
    "MP": {
        "full_name": "Mathematical Programming",
        "issn": ["0025-5610", "1436-4646"],
        "openalex_id": "S34536029",
    },
    "EJOR": {
        "full_name": "European Journal of Operational Research",
        "issn": ["0377-2217"],
        "openalex_id": "S119206744",
    },
    "COR": {
        "full_name": "Computers & Operations Research",
        "issn": ["0305-0548"],
        "openalex_id": "S173256270",
    },
    "ORL": {
        "full_name": "Operations Research Letters",
        "issn": ["0167-6377", "1872-7468"],
        "openalex_id": "S27769002",
    },
    "AOR": {
        "full_name": "Annals of Operations Research",
        "issn": ["0254-5330", "1572-9338"],
        "openalex_id": "S57667410",
    },
}


# ---------------------------------------------------------------------------
#  OpenAlex API helpers
# ---------------------------------------------------------------------------

OPENALEX_API = "https://api.openalex.org"
# Polite pool: add email for faster rate limits (optional)
MAILTO = ""  # Set your email here for higher rate limits

def _api_get(url: str, params: dict = None) -> dict:
    """Make a GET request to OpenAlex API with rate limiting."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    if MAILTO:
        sep = "&" if "?" in url else "?"
        url = url + sep + f"mailto={MAILTO}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def fetch_papers_for_journal(
    journal_key: str,
    journal_info: dict,
    from_date: str,
    to_date: str,
    max_papers: int = 200,
) -> list[dict]:
    """Fetch papers from a single journal via OpenAlex API."""
    source_id = journal_info.get("openalex_id", "")
    papers = []
    cursor = "*"
    per_page = min(max_papers, 200)

    while len(papers) < max_papers:
        params = {
            "filter": f"primary_location.source.id:https://openalex.org/{source_id},"
                      f"from_publication_date:{from_date},"
                      f"to_publication_date:{to_date},"
                      f"type:article",
            "sort": "publication_date:desc",
            "per_page": per_page,
            "cursor": cursor,
        }

        data = _api_get(f"{OPENALEX_API}/works", params)
        results = data.get("results", [])
        if not results:
            break

        for work in results:
            if len(papers) >= max_papers:
                break

            # Extract key fields
            title = work.get("title", "")
            doi = work.get("doi", "")
            pub_date = work.get("publication_date", "")
            abstract_inv = work.get("abstract_inverted_index") or {}

            # Reconstruct abstract from inverted index
            abstract = _reconstruct_abstract(abstract_inv)

            # Authors
            authorships = work.get("authorships", [])
            authors = []
            for a in authorships:
                author = a.get("author", {})
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

            # Open access PDF URL
            oa = work.get("open_access", {})
            pdf_url = oa.get("oa_url", "")
            is_oa = oa.get("is_oa", False)

            # Also check primary_location for PDF
            primary_loc = work.get("primary_location", {})
            if not pdf_url and primary_loc:
                pdf_url = (primary_loc.get("pdf_url") or "")

            # Concepts/topics
            concepts = []
            for c in work.get("concepts", [])[:5]:
                concepts.append(c.get("display_name", ""))

            papers.append({
                "title": title,
                "authors": authors,
                "publication_date": pub_date,
                "doi": doi,
                "journal": journal_info["full_name"],
                "journal_key": journal_key,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "is_open_access": is_oa,
                "openalex_id": work.get("id", ""),
                "concepts": concepts,
                "cited_by_count": work.get("cited_by_count", 0),
            })

        # Pagination
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        if not cursor:
            break

        time.sleep(0.1)  # Be polite

    return papers


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    # {word: [positions]} -> sorted by position
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


# ---------------------------------------------------------------------------
#  PDF download
# ---------------------------------------------------------------------------

def download_pdf(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download a PDF from URL. Returns True on success."""
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "ORBench/1.0 (research; mailto:orbench@example.com)",
            "Accept": "application/pdf",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            # Basic check: PDF should start with %PDF
            if data[:4] == b"%PDF" or "pdf" in content_type.lower():
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(data)
                return True
    except Exception:
        pass
    return False


def sanitize_filename(s: str, max_len: int = 50) -> str:
    """Make a string safe for use as a filename."""
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s[:max_len]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch recent papers from major OR journals (via OpenAlex API)"
    )
    parser.add_argument(
        "--months", type=int, default=12,
        help="How many months back to search (default: 12)"
    )
    parser.add_argument(
        "--journals", type=str, default=None,
        help=f"Comma-separated journal keys to include (default: all). "
             f"Available: {','.join(OR_JOURNALS.keys())}"
    )
    parser.add_argument(
        "--max-per-journal", type=int, default=200,
        help="Max papers per journal (default: 200)"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download open-access PDFs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="Agent/Paper/collected",
        help="Output directory (default: Agent/Paper/collected)"
    )
    parser.add_argument(
        "--mailto", type=str, default="",
        help="Email for OpenAlex polite pool (faster rate limits)"
    )
    args = parser.parse_args()

    global MAILTO
    if args.mailto:
        MAILTO = args.mailto

    # Date range
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")

    # Select journals
    if args.journals:
        selected_keys = [k.strip().upper() for k in args.journals.split(",")]
        journals = {k: OR_JOURNALS[k] for k in selected_keys if k in OR_JOURNALS}
        unknown = [k for k in selected_keys if k not in OR_JOURNALS]
        if unknown:
            print(f"WARNING: Unknown journal keys: {unknown}")
            print(f"Available: {list(OR_JOURNALS.keys())}")
    else:
        journals = OR_JOURNALS

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching papers from {len(journals)} journals")
    print(f"Date range: {from_date} to {to_date}")
    print(f"Max per journal: {args.max_per_journal}")
    print(f"Output: {out_dir}")
    print()

    all_papers = []
    for key, info in journals.items():
        print(f"  [{key}] {info['full_name']}...", end=" ", flush=True)
        try:
            papers = fetch_papers_for_journal(
                key, info, from_date, to_date, args.max_per_journal
            )
            all_papers.extend(papers)
            print(f"{len(papers)} papers")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nTotal: {len(all_papers)} papers")

    # Save metadata JSON
    meta_path = out_dir / "papers_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {meta_path}")

    # Save CSV summary
    csv_path = out_dir / "papers_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "journal", "date", "title", "authors", "doi",
            "is_oa", "cited_by", "concepts", "abstract_preview"
        ])
        for p in all_papers:
            writer.writerow([
                p["journal_key"],
                p["publication_date"],
                p["title"],
                "; ".join(p["authors"][:3]) + ("..." if len(p["authors"]) > 3 else ""),
                p["doi"],
                p["is_open_access"],
                p["cited_by_count"],
                "; ".join(p["concepts"][:3]),
                p["abstract"][:150] + "..." if len(p["abstract"]) > 150 else p["abstract"],
            ])
    print(f"CSV saved to: {csv_path}")

    # Download PDFs
    if args.download:
        pdf_dir = out_dir / "pdfs"
        pdf_dir.mkdir(exist_ok=True)
        oa_papers = [p for p in all_papers if p["pdf_url"]]
        print(f"\nDownloading {len(oa_papers)} open-access PDFs...")

        downloaded = 0
        for i, p in enumerate(oa_papers):
            first_author = sanitize_filename(p["authors"][0].split()[-1]) if p["authors"] else "unknown"
            year = p["publication_date"][:4] if p["publication_date"] else "0000"
            fname = f"{p['journal_key']}_{year}_{first_author}_{sanitize_filename(p['title'], 30)}.pdf"
            save_path = pdf_dir / fname

            if save_path.exists():
                downloaded += 1
                continue

            ok = download_pdf(p["pdf_url"], save_path)
            if ok:
                downloaded += 1
                print(f"  [{downloaded}/{len(oa_papers)}] {fname}")
            else:
                print(f"  [SKIP] {p['title'][:60]}... (download failed)")

            time.sleep(0.2)

        print(f"\nDownloaded {downloaded}/{len(oa_papers)} PDFs to {pdf_dir}")

    # Print journal-level stats
    print(f"\n{'='*60}")
    print(f"  Summary by journal")
    print(f"{'='*60}")
    from collections import Counter
    journal_counts = Counter(p["journal_key"] for p in all_papers)
    oa_counts = Counter(p["journal_key"] for p in all_papers if p["is_open_access"])
    for key in sorted(journal_counts.keys()):
        total = journal_counts[key]
        oa = oa_counts.get(key, 0)
        print(f"  {key:6s} {OR_JOURNALS[key]['full_name']:50s} {total:4d} papers ({oa} OA)")
    print(f"  {'TOTAL':6s} {'':50s} {len(all_papers):4d} papers")


if __name__ == "__main__":
    main()
