"""
Build a pathology concept set from:
  1. NCIt (NCI Thesaurus) via EVS REST API — tumor types, morphology, cell types
  2. HPO (Human Phenotype Ontology) — site-specific neoplasm vocabulary
  3. A hand-curated list of H&E-visible morphological features

Output: data/pathology_concepts.txt — one concept per line, lowercase.
Apply a prompt template at encoding time, e.g.:
    texts = [f"histopathology image showing {c}" for c in concepts]

Usage:
    python build_concept_set.py                        # full run (~15 min)
    python build_concept_set.py --skip-branches        # curated list only (instant)
    python build_concept_set.py --skip-hpo             # NCIt + curated only
    python build_concept_set.py --output data/custom.txt --max-words 6
"""

import argparse
import io
import time
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NCIt (EVS REST API)
# ---------------------------------------------------------------------------

EVS_BASE = "https://api-evsrest.nci.nih.gov/api/v1/concept/ncit"

DEFAULT_BRANCHES = {
    # Tumor types (histology-based)
    "C4741":  "Neoplasm by Morphology",       # superset: carcinoma, sarcoma, lymphoma, ...
    # Morphological patterns and tissue features
    "C35886": "Morphologic Architectural Pattern",  # acinar, papillary, cribriform, ...
    "C36184": "Necrosis",
    "C3044":  "Fibrosis",
    "C3137":  "Inflammation",
    # Cell types
    "C12508": "Cell",
}

# Trusted synonym sources for NCIt
NCIT_SYN_SOURCES = {"", "NCI", "CDISC", "FDA", "NICHD"}

# ---------------------------------------------------------------------------
# HPO (Human Phenotype Ontology OBO)
# ---------------------------------------------------------------------------

HPO_OBO_URL = (
    "https://raw.githubusercontent.com/obophenotype/"
    "human-phenotype-ontology/master/hp.obo"
)

# HPO roots to include descendants from
HPO_ROOTS = {
    "HP:0002664": "Neoplasm",               # 985 descendants, site-specific tumor types
}

# Only include EXACT synonyms from HPO (not layperson, abbreviation, etc.)
HPO_EXACT_TYPES = {"EXACT", ""}

# ---------------------------------------------------------------------------
# Hand-curated morphological concepts (H&E-visible features)
# These match CONCH's caption training distribution better than single words.
# ---------------------------------------------------------------------------

CURATED_CONCEPTS = [
    # --- Nuclear features ---
    "nuclear pleomorphism",
    "nuclear atypia",
    "nuclear enlargement",
    "nuclear irregularity",
    "vesicular nucleus",
    "hyperchromatic nucleus",
    "prominent nucleolus",
    "high nuclear-to-cytoplasmic ratio",
    "nuclear grooves",
    "nuclear pseudoinclusions",
    "optically clear nucleus",
    # --- Mitotic activity ---
    "mitotic figure",
    "atypical mitosis",
    "increased mitotic activity",
    "apoptotic body",
    # --- Cytoplasmic features ---
    "eosinophilic cytoplasm",
    "basophilic cytoplasm",
    "amphophilic cytoplasm",
    "clear cytoplasm",
    "granular cytoplasm",
    "vacuolated cytoplasm",
    "abundant cytoplasm",
    "scant cytoplasm",
    # --- Cell types ---
    "signet ring cell",
    "clear cell",
    "spindle cell",
    "foam cell",
    "plasma cell",
    "multinucleated giant cell",
    "tumor giant cell",
    "myoepithelial cell",
    "goblet cell",
    "neuroendocrine cell",
    "squamous cell",
    "columnar cell",
    "cuboidal cell",
    # --- Inflammatory infiltrate ---
    "tumor infiltrating lymphocytes",
    "lymphocytic infiltrate",
    "eosinophilic infiltrate",
    "neutrophilic infiltrate",
    "plasma cell infiltrate",
    "lymphoid aggregate",
    "germinal center",
    "chronic inflammation",
    "acute inflammation",
    "granulomatous inflammation",
    "foreign body giant cell reaction",
    # --- Architectural patterns ---
    "papillary pattern",
    "cribriform pattern",
    "acinar pattern",
    "solid pattern",
    "micropapillary pattern",
    "tubular pattern",
    "glandular pattern",
    "sheet-like pattern",
    "single file pattern",
    "trabecular pattern",
    "alveolar pattern",
    "comedo pattern",
    "lepidic pattern",
    "invasive front",
    "back-to-back glands",
    "complex glandular architecture",
    "irregular glandular architecture",
    "fused glands",
    "angulated glands",
    # --- Stromal features ---
    "desmoplastic stroma",
    "myxoid stroma",
    "hyalinized stroma",
    "fibrous stroma",
    "tumor stroma",
    "peritumoral fibrosis",
    "stromal invasion",
    "reactive stroma",
    "scirrhous stroma",
    # --- Vascular and perineural ---
    "lymphovascular invasion",
    "perineural invasion",
    "vascular invasion",
    "angiolymphatic invasion",
    "blood vessel invasion",
    # --- Necrosis ---
    "tumor necrosis",
    "comedo necrosis",
    "geographic necrosis",
    "coagulative necrosis",
    "central necrosis",
    "pseudopalisading necrosis",
    # --- Secretory and mucin features ---
    "mucin production",
    "extracellular mucin",
    "intracellular mucin",
    "mucin pool",
    "signet ring cell differentiation",
    "gland formation",
    # --- Differentiation and grade ---
    "poorly differentiated",
    "moderately differentiated",
    "well differentiated",
    "high grade",
    "low grade",
    "intermediate grade",
    # --- Invasion and spread ---
    "invasive carcinoma",
    "in situ carcinoma",
    "microinvasion",
    "capsular invasion",
    "stromal microinvasion",
    "lymph node metastasis",
    "tumor budding",
    "satellite nodule",
    # --- Special morphological features ---
    "keratinization",
    "intercellular bridges",
    "keratin pearl",
    "psammoma body",
    "calcification",
    "hemosiderin deposition",
    "basement membrane",
    "retraction artifact",
    "crush artifact",
    "squamous differentiation",
    "glandular differentiation",
    "neuroendocrine differentiation",
    "rhabdoid differentiation",
    "sarcomatoid differentiation",
    # --- Tissue compartments ---
    "tumor epithelium",
    "peritumoral stroma",
    "tumor microenvironment",
    "normal epithelium",
    "lamina propria",
    "muscularis propria",
    "submucosa",
    "surgical margin",
    # --- Grading systems ---
    "Gleason pattern 3",
    "Gleason pattern 4",
    "Gleason pattern 5",
    "nuclear grade 1",
    "nuclear grade 2",
    "nuclear grade 3",
    "Scarff-Bloom-Richardson grade",
]

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

BLOCKLIST = {
    "cell", "neoplasm", "disease", "disorder", "condition", "syndrome",
    "finding", "abnormality", "process", "procedure", "type", "stage",
    "grade", "category", "group", "entity", "unspecified",
    "not otherwise specified", "nos", "other", "unknown", "related",
}

SKIP_SUBSTRINGS = ["[", "]", "(", ")", "\\", "_", "–"]


def is_valid(term: str, min_words: int, max_words: int) -> bool:
    term = term.strip().lower()
    if not term:
        return False
    if any(c in term for c in SKIP_SUBSTRINGS):
        return False
    words = term.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if term in BLOCKLIST:
        return False
    # skip NCIt concept codes like "c12345"
    if any(w[0] == "c" and w[1:].isdigit() and len(w) > 3 for w in words):
        return False
    return True


# ---------------------------------------------------------------------------
# NCIt helpers
# ---------------------------------------------------------------------------

def ncit_get_descendants(code: str, page_size: int = 200, delay: float = 0.25) -> list[dict]:
    concepts, from_record = [], 0
    pbar = tqdm(desc=f"  Fetching {code}", unit="concepts", leave=False)
    while True:
        resp = requests.get(
            f"{EVS_BASE}/{code}/descendants",
            params={"include": "synonyms", "pageSize": page_size, "fromRecord": from_record},
            timeout=30,
        )
        if resp.status_code == 404:
            print(f"  WARNING: {code} not found, skipping")
            break
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        concepts.extend(batch)
        pbar.update(len(batch))
        if len(batch) < page_size:
            break
        from_record += page_size
        time.sleep(delay)
    pbar.close()
    return concepts


def ncit_extract_names(concept: dict, include_synonyms: bool) -> list[str]:
    names = [concept.get("name", "")]
    if include_synonyms:
        for syn in concept.get("synonyms", []):
            if syn.get("source", "") in NCIT_SYN_SOURCES:
                names.append(syn.get("name", ""))
    return [n for n in names if n]


# ---------------------------------------------------------------------------
# HPO OBO helpers
# ---------------------------------------------------------------------------

def hpo_download_and_parse(url: str) -> tuple[dict, dict]:
    """
    Returns:
        terms   : {hp_id: {'name': str, 'synonyms': [(text, type)], 'obsolete': bool}}
        children: {hp_id: [child_hp_id, ...]}
    """
    print("Downloading HPO OBO file...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    terms: dict = {}
    current: dict = {}

    for line in io.StringIO(resp.text):
        line = line.rstrip()
        if line == "[Term]":
            if current.get("id"):
                terms[current["id"]] = current
            current = {}
        elif line.startswith("id: "):
            current["id"] = line[4:].strip()
        elif line.startswith("name: "):
            current["name"] = line[6:].strip()
        elif line.startswith("synonym: "):
            parts = line[9:].split('"')
            if len(parts) >= 3:
                syn_text = parts[1]
                syn_type_raw = parts[2].strip().split()
                syn_type = syn_type_raw[0] if syn_type_raw else ""
                current.setdefault("synonyms", []).append((syn_text, syn_type))
        elif line.startswith("is_a: "):
            parent = line[6:].split("!")[0].strip()
            current.setdefault("is_a", []).append(parent)
        elif line.startswith("is_obsolete: true"):
            current["obsolete"] = True

    if current.get("id"):
        terms[current["id"]] = current

    # Build children map
    children: dict = {tid: [] for tid in terms}
    for tid, t in terms.items():
        for parent in t.get("is_a", []):
            if parent in children:
                children[parent].append(tid)

    print(f"  Parsed {len(terms)} HPO terms")
    return terms, children


def hpo_get_descendants(root: str, terms: dict, children: dict) -> list[str]:
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            if not terms.get(child, {}).get("obsolete"):
                result.append(child)
                stack.append(child)
    return result


def hpo_extract_names(term_dict: dict, include_synonyms: bool) -> list[str]:
    names = [term_dict.get("name", "")]
    if include_synonyms:
        for text, syn_type in term_dict.get("synonyms", []):
            if syn_type in HPO_EXACT_TYPES:
                names.append(text)
    return [n for n in names if n]


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_concept_set(args) -> list[str]:
    all_terms: set[str] = set()

    # 1. Curated list
    for concept in CURATED_CONCEPTS:
        term = concept.strip().lower()
        if term:
            all_terms.add(term)
    print(f"Curated concepts: {len(all_terms)}")

    if args.skip_branches:
        return sorted(all_terms)

    # 2. NCIt branches
    branches = {code: DEFAULT_BRANCHES.get(code, code) for code in args.branches}

    for code, label in branches.items():
        print(f"\nNCIt {code} ({label})...")
        concepts = ncit_get_descendants(code)
        print(f"  {len(concepts)} concepts retrieved")
        before = len(all_terms)
        for concept in concepts:
            for name in ncit_extract_names(concept, not args.no_synonyms):
                term = name.strip().lower()
                if is_valid(term, args.min_words, args.max_words):
                    all_terms.add(term)
        print(f"  +{len(all_terms) - before} new terms (total: {len(all_terms)})")

    # 3. HPO
    if not args.skip_hpo:
        print()
        hpo_terms, hpo_children = hpo_download_and_parse(HPO_OBO_URL)
        for root_id, root_label in HPO_ROOTS.items():
            desc_ids = hpo_get_descendants(root_id, hpo_terms, hpo_children)
            print(f"HPO {root_id} ({root_label}): {len(desc_ids)} descendants")
            before = len(all_terms)
            for hpid in desc_ids:
                t = hpo_terms.get(hpid, {})
                for name in hpo_extract_names(t, not args.no_synonyms):
                    term = name.strip().lower()
                    if is_valid(term, args.min_words, args.max_words):
                        all_terms.add(term)
            print(f"  +{len(all_terms) - before} new terms (total: {len(all_terms)})")

    return sorted(all_terms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build NCIt + HPO pathology concept set for CONCH-Dissect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--branches", nargs="+", default=list(DEFAULT_BRANCHES.keys()), metavar="CODE",
        help="NCIt codes to fetch descendants from",
    )
    parser.add_argument("--output", default="data/pathology_concepts.txt")
    parser.add_argument(
        "--min-words", type=int, default=2,
        help="Minimum word count per concept",
    )
    parser.add_argument(
        "--max-words", type=int, default=8,
        help="Maximum word count per concept",
    )
    parser.add_argument("--no-synonyms", action="store_true")
    parser.add_argument("--skip-branches", action="store_true",
                        help="Output curated list only, skip all API calls")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip HPO download")
    args = parser.parse_args()

    print(f"Output: {args.output}")
    print(f"Branches: {args.branches}")
    print(f"Word range: {args.min_words}–{args.max_words} | Synonyms: {not args.no_synonyms}")
    print()

    concepts = build_concept_set(args)

    with open(args.output, "w") as f:
        f.write("\n".join(concepts))

    print(f"\nTotal: {len(concepts)} concepts -> {args.output}")

    import random
    print("\nRandom sample (20):")
    for c in sorted(random.sample(concepts, min(20, len(concepts)))):
        print(f"  {c}")

    print(
        "\n--- Encoding note ---\n"
        "Use a prompt template at encoding time (do NOT bake it into this file):\n"
        '  texts = [f"histopathology image showing {c}" for c in concepts]\n'
        "Ensembling two templates further boosts CONCH zero-shot performance:\n"
        '  texts = [f"histopathology image showing {c}" for c in concepts]\n'
        '         + [f"a pathological finding of {c}" for c in concepts]'
    )


if __name__ == "__main__":
    main()
