"""
Build a pathology concept set from OpenI/NLM figure data embedded in the PLIP CSV.

Two sources:
  1. Keywords embedded in OpenI image URLs (?keywords=...)        — 548 entries
  2. Figure captions fetched from PubMed Central via NCBI efetch  — remaining PMC IDs

Output: concept_sets/openi_concepts.txt
"""

import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd

CSV_PATH  = "/home/maracuja/dl/dataframe_208K_rows.csv"
OUT_PATH  = Path(__file__).parent / "openi_concepts.txt"
NCBI_RATE = 0.4   # seconds between NCBI requests (max 3/s without API key)

# ── load OpenI rows ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, header=None)
openi = df[df[3].str.contains("openi.nlm.nih.gov", na=False)].copy()
print(f"OpenI rows: {len(openi)}")

concepts = set()

# ── 1. extract inline keywords ────────────────────────────────────────────────
def clean(term):
    term = term.strip().lower()
    # skip very short or generic terms
    if len(term) < 4:
        return None
    if term in {"pale", "stain", "tissue", "cells", "cell", "image", "figure"}:
        return None
    return term

keyword_count = 0
for url in openi[3]:
    kw_str = parse_qs(urlparse(url).query).get("keywords", [""])[0]
    if not kw_str:
        continue
    for term in kw_str.split(","):
        c = clean(term)
        if c:
            concepts.add(c)
    keyword_count += 1

print(f"Concepts from inline keywords ({keyword_count} URLs): {len(concepts)}")

# ── 2. fetch PMC figure captions ──────────────────────────────────────────────
pmc_re = re.compile(r"PMC(\d+)_")
pmc_ids = []
for url in openi[3]:
    m = pmc_re.search(url)
    if m:
        pmc_ids.append(m.group(1))
pmc_ids = sorted(set(pmc_ids))
print(f"Unique PMC IDs to fetch: {len(pmc_ids)}")

EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pmc&id={pmc_id}&rettype=full&retmode=xml"
)

def fetch_captions(pmc_id):
    """Return list of figure caption strings from a PMC article."""
    url = EFETCH.format(pmc_id=pmc_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        xml_bytes = resp.read()
    except Exception as e:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    captions = []
    for cap in root.iter("caption"):
        text = " ".join(cap.itertext()).strip()
        text = re.sub(r"\s+", " ", text)
        if 20 < len(text) < 400:
            captions.append(text.lower())
    return captions

caption_concepts = []
failed = 0
for i, pmc_id in enumerate(pmc_ids):
    caps = fetch_captions(pmc_id)
    caption_concepts.extend(caps)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(pmc_ids)} PMC IDs fetched, "
              f"{len(caption_concepts)} captions so far, {failed} failed")
    time.sleep(NCBI_RATE)

print(f"Raw captions from PMC: {len(caption_concepts)}")

# Add captions to concept set (full sentences, deduplicated)
for cap in caption_concepts:
    cap = cap.strip()
    if cap:
        concepts.add(cap)

# ── write output ──────────────────────────────────────────────────────────────
concepts_sorted = sorted(concepts)
OUT_PATH.write_text("\n".join(concepts_sorted) + "\n")
print(f"\nSaved {len(concepts_sorted)} concepts → {OUT_PATH}")
