"""
Extract pathology concepts from TCGA diagnostic reports.

Strategy:
  1. Extract values from structured fields (Histologic type, grade, etc.)
  2. Extract 2-5-gram noun phrases from DIAGNOSIS / FINAL DIAGNOSIS free text
  3. Keep phrases appearing in >= MIN_FREQ reports (filters noise and one-offs)
  4. Save data/tcga_concepts.txt  (TCGA-only)
  5. Save data/pathology_concepts_combined.txt (TCGA + NCIt/HPO merged)

Usage:
    python build_tcga_concepts.py
    python build_tcga_concepts.py --reports /path/to/TCGA_Reports.csv
    python build_tcga_concepts.py --min-freq 5 --max-words 7
    python build_tcga_concepts.py --no-combine   # skip building combined file
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Structured field extraction
# ---------------------------------------------------------------------------

# Field names to extract values from directly (case-insensitive)
STRUCTURED_FIELDS = [
    r"histologic\s+type",
    r"histological\s+type",
    r"tumor\s+type",
    r"histologic\s+grade",
    r"histological\s+grade",
    r"nuclear\s+grade",
    r"additional\s+pathologic\s+findings",
    r"additional\s+findings",
    r"microscopic\s+description",
    r"lymphovascular\s+invasion",
    r"perineural\s+invasion",
    r"vascular\s+invasion",
    r"margin\s+status",
    r"surgical\s+margins",
]

FIELD_PAT = re.compile(
    r"(?:" + "|".join(STRUCTURED_FIELDS) + r")\s*:\s*([^\n.;]{5,200})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Diagnosis section extraction
# ---------------------------------------------------------------------------

# Markers that start a diagnosis section
DIAG_START = re.compile(
    r"(?:FINAL\s+DIAGNOSIS|DIAGNOSIS|PRIMARY\s+DIAGNOSIS|PATHOLOGIC\s+DIAGNOSIS)"
    r"\s*[:\n]",
    re.IGNORECASE,
)

# Markers that end a diagnosis section (start of gross description etc.)
DIAG_END = re.compile(
    r"(?:GROSS\s+DESCRIPTION|GROSS\s+PATHOLOGY|MACROSCOPIC|CLINICAL\s+HISTORY"
    r"|SUMMARY\s+OF\s+SECTIONS|BLOCK\s+SUMMARY|SECTION\s+CODE|INTRAOPERATIVE)",
    re.IGNORECASE,
)

def extract_diagnosis_text(text: str) -> str:
    """Return the DIAGNOSIS section(s) only, stripped of admin boilerplate."""
    parts = []
    for m in DIAG_START.finditer(text):
        start = m.end()
        end_m = DIAG_END.search(text, start)
        end = end_m.start() if end_m else start + 800  # cap at 800 chars
        parts.append(text[start:end])
    return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# N-gram extraction from free text
# ---------------------------------------------------------------------------

# Tokens to skip entirely
STOPWORDS = {
    # Articles, prepositions, conjunctions
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "but", "is", "are", "was", "were", "be", "been", "not", "no",
    "by", "from", "as", "this", "that", "it", "its", "into", "within",
    "without", "per", "than", "has", "have", "had", "all", "also", "more",
    "most", "each", "both", "other", "which", "who", "if", "when", "where",
    # Measurement / quantity
    "cm", "mm", "x", "approximately", "measured", "measuring", "measures",
    "greatest", "dimension", "diameter", "weight", "size", "maximal",
    "maximum", "minimum", "total", "number", "level", "grade", "stage",
    # Specimen / procedure admin
    "submitted", "received", "identified", "noted", "seen", "present",
    "specimen", "section", "block", "slide", "case", "report", "page",
    "summary", "procedure", "biopsy", "excision", "resection", "labeled",
    "consistent", "compatible", "representative", "include", "including",
    "part", "right", "left", "upper", "lower", "lateral", "medial",
    "anterior", "posterior", "date", "patient", "history", "clinical",
    "please", "see", "note", "above", "below", "status", "addendum",
    "addenda", "stand", "alone", "certify", "certified", "laboratory",
    "protocol", "pathologic", "pathological", "procedures", "temporal",
    # Staging/coding jargon
    "ajcc", "tnm", "nci", "figo", "who", "icd",
    # Common anatomical prepositions used as filler
    "involving", "arising", "extending", "adjacent", "associated",
}

# Only allow tokens that look like real words (no pure numbers, codes, etc.)
WORD_PAT = re.compile(r"^[a-z][a-z\-']{1,}$")

# Blocklist phrases that are administrative, not visual pathology concepts
PHRASE_BLOCKLIST = {
    # Administrative / non-visual
    "not identified", "not present", "not seen", "no evidence",
    "free of tumor", "free of carcinoma", "uninvolved by",
    "cannot be", "could not be", "unable to", "within normal limits",
    "representative sections", "sections submitted", "specimen received",
    "gross description", "frozen section", "final diagnosis",
    "permanent diagnosis", "benign margin", "number of", "total number",
    "personal examination", "reviewed and approved", "see comment",
    "above diagnosis", "clinical history", "same diagnosis",
    "diagnosis same", "greatest dimension", "tumor size",
    "frozen diagnosis", "sections sections", "tumor tumor",
    # Anatomy without pathological meaning
    "lymph node", "lymph nodes", "fallopian tube", "soft tissue",
    "salpingo oophorectomy", "pelvic lymph", "regional lymph",
    "iliac lymph", "one lymph", "one lymph node", "benign lymph",
    # Measurement/staging boilerplate
    "nodes examined", "nodes negative", "negative malignancy",
    "negative tumor", "free tumor", "margins free",
}

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, filter stopwords."""
    tokens = re.sub(r"[^a-zA-Z\-' ]", " ", text).lower().split()
    return [t for t in tokens if WORD_PAT.match(t) and t not in STOPWORDS and len(t) > 2]


def extract_ngrams(tokens: list[str], min_n: int, max_n: int) -> list[str]:
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i+n])
            ngrams.append(gram)
    return ngrams


def split_into_sentences(text: str) -> list[str]:
    """Split on sentence/clause boundaries to prevent n-grams crossing them."""
    return re.split(r"[.;:\n]", text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SKIP_SUBSTRINGS = ["[", "]", "\\", "_", "(", ")", "/", ":", "%"]

def is_valid_concept(term: str, min_words: int, max_words: int) -> bool:
    term = term.strip()
    if not term:
        return False
    if any(c in term for c in SKIP_SUBSTRINGS):
        return False
    words = term.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if term in PHRASE_BLOCKLIST:
        return False
    # Skip if starts/ends with a stopword (usually incomplete phrases)
    if words[0] in STOPWORDS or words[-1] in STOPWORDS:
        return False
    # Skip pure numeric or code-like tokens
    if any(w.isdigit() or re.match(r'^p[ttnm]\d', w) for w in words):
        return False
    return True


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_concepts(df: pd.DataFrame, min_freq: int, min_words: int, max_words: int) -> list[str]:
    structured_terms: Counter = Counter()
    ngram_counts: Counter = Counter()

    for text in tqdm(df["text"].dropna(), desc="Processing reports"):
        # 1. Structured field values
        for m in FIELD_PAT.finditer(text):
            val = m.group(1).strip().lower()
            # Split on conjunctions like "and", semicolons, hyphens used as separators
            for part in re.split(r"\band\b|;|,\s*(?=[a-z])", val):
                part = part.strip(" .-")
                if is_valid_concept(part, min_words, max_words):
                    structured_terms[part] += 1

        # 2. N-grams from diagnosis free text — split into sentences first
        # to prevent phrases crossing clause/sentence boundaries
        diag_text = extract_diagnosis_text(text)
        if diag_text:
            for sentence in split_into_sentences(diag_text):
                tokens = tokenize(sentence)
                if len(tokens) < 2:
                    continue
                for gram in extract_ngrams(tokens, min_words, max_words):
                    if is_valid_concept(gram, min_words, max_words):
                        ngram_counts[gram] += 1

    # Structured fields: lower threshold (min_freq // 2, min 2)
    structured_threshold = max(2, min_freq // 2)
    kept_structured = {t for t, c in structured_terms.items() if c >= structured_threshold}

    # N-grams: apply full min_freq threshold
    kept_ngrams = {t for t, c in ngram_counts.items() if c >= min_freq}

    all_concepts = kept_structured | kept_ngrams

    # Post-filter: remove phrases in PHRASE_BLOCKLIST
    all_concepts = {c for c in all_concepts if c not in PHRASE_BLOCKLIST}

    print(f"\nStructured fields: {len(kept_structured)} concepts (threshold >= {structured_threshold})")
    print(f"N-gram phrases:    {len(kept_ngrams)} concepts (threshold >= {min_freq})")
    print(f"Combined unique:   {len(all_concepts)}")

    # Show top structured and n-gram terms by frequency for inspection
    print("\nTop 20 structured field values:")
    for t, c in structured_terms.most_common(20):
        if t in kept_structured:
            print(f"  {c:5d}  {t}")

    print("\nTop 20 n-gram phrases:")
    for t, c in ngram_counts.most_common(20):
        if t in kept_ngrams:
            print(f"  {c:5d}  {t}")

    return sorted(all_concepts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract pathology concepts from TCGA reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports",
        default="/home/maracuja/projects/conch-dissect/TCGA_Reports.csv",
    )
    parser.add_argument("--tcga-output", default="data/tcga_concepts.txt")
    parser.add_argument(
        "--combined-output", default="data/pathology_concepts_combined.txt"
    )
    parser.add_argument(
        "--ncit-file",
        default="data/pathology_concepts.txt",
        help="Existing NCIt/HPO concept file to merge for the combined output",
    )
    parser.add_argument(
        "--min-freq", type=int, default=50,
        help="Minimum number of reports a phrase must appear in (n-grams only; "
             "structured fields use min-freq // 2)",
    )
    parser.add_argument(
        "--min-words", type=int, default=2,
        help="Minimum words per concept",
    )
    parser.add_argument(
        "--max-words", type=int, default=7,
        help="Maximum words per concept",
    )
    parser.add_argument(
        "--no-combine", action="store_true",
        help="Skip writing the combined output file",
    )
    args = parser.parse_args()

    print(f"Loading {args.reports}...")
    df = pd.read_csv(args.reports)
    print(f"  {len(df)} reports loaded")

    tcga_concepts = extract_concepts(
        df,
        min_freq=args.min_freq,
        min_words=args.min_words,
        max_words=args.max_words,
    )

    # Save TCGA-only
    Path(args.tcga_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.tcga_output, "w") as f:
        f.write("\n".join(tcga_concepts))
    print(f"\nSaved {len(tcga_concepts)} TCGA concepts -> {args.tcga_output}")

    # Save combined
    if not args.no_combine:
        ncit_path = Path(args.ncit_file)
        if ncit_path.exists():
            with open(ncit_path) as f:
                ncit_concepts = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(ncit_concepts)} NCIt/HPO concepts from {args.ncit_file}")
        else:
            print(f"WARNING: {args.ncit_file} not found — combined file will be TCGA-only")
            ncit_concepts = set()

        combined = sorted(set(tcga_concepts) | ncit_concepts)
        with open(args.combined_output, "w") as f:
            f.write("\n".join(combined))

        tcga_only = len(set(tcga_concepts) - ncit_concepts)
        ncit_only = len(ncit_concepts - set(tcga_concepts))
        overlap = len(set(tcga_concepts) & ncit_concepts)
        print(f"\nCombined concept set:")
        print(f"  TCGA only:    {tcga_only}")
        print(f"  NCIt/HPO only:{ncit_only}")
        print(f"  Overlap:      {overlap}")
        print(f"  Total:        {len(combined)}")
        print(f"Saved -> {args.combined_output}")

    import random
    print(f"\nRandom sample from TCGA concepts:")
    for c in sorted(random.sample(tcga_concepts, min(20, len(tcga_concepts)))):
        print(f"  {c}")


if __name__ == "__main__":
    main()
