"""
Build three additional concept sets from all_concepts.txt:

1. all_concepts_moderate_dedup.txt
   - Keeps one AJCC version per concept (prefer v8, else highest)
   - Keeps core term + "metastatic X" only (drops advanced/recurrent/refractory/etc.)

2. all_concepts_aggressive_dedup.txt
   - Strips all AJCC versioning (keep one per disease)
   - Keeps only the bare core term (drops all status prefixes)

3. curated_concepts.txt
   - Small (~200), hand-defined, clean set covering:
     * 10 TCGA cancer types (tissue-specific morphology)
     * Staging morphology (invasion, grade, margins, nodes)
     * General histomorphology markers
"""

import re
from pathlib import Path

SRC = Path(__file__).parent / "all_concepts.txt"

with open(SRC) as f:
    concepts = [l.strip() for l in f if l.strip()]

print(f"Source: {len(concepts)} concepts")

STATUS_PREFIXES = [
    "advanced ", "locally advanced ", "metastatic ", "recurrent ",
    "refractory ", "unresectable ", "relapsed ", "progressive ",
    "newly diagnosed ", "previously untreated ", "previously treated ",
]

AJCC_RE = re.compile(r"\s+ajcc\s+v\d+", re.I)

def strip_status(c):
    for pfx in STATUS_PREFIXES:
        if c.lower().startswith(pfx):
            return c[len(pfx):]
    return c

def strip_ajcc(c):
    return AJCC_RE.sub("", c).strip()


# ── 1. Moderate dedup ─────────────────────────────────────────────────────────
# For AJCC: keep v8 if present, else v9, v7, v6 — one per base concept
# For status prefixes: keep bare core + metastatic variant only

def moderate_dedup(concepts):
    # Step 1: AJCC — group by base (stripped of ajcc version), keep preferred
    ajcc_groups = {}  # base -> {version: full_concept}
    non_ajcc = []
    for c in concepts:
        m = re.search(r"ajcc\s+v(\d+)", c, re.I)
        if m:
            base = AJCC_RE.sub("", c).strip()
            ver = int(m.group(1))
            if base not in ajcc_groups or ver > ajcc_groups[base][0]:
                ajcc_groups[base] = (ver, c)
        else:
            non_ajcc.append(c)
    ajcc_kept = [v for _, v in ajcc_groups.values()]

    # Step 2: status prefixes — group by core, keep core + metastatic only
    core_seen = set()
    result = []
    for c in ajcc_kept + non_ajcc:
        core = strip_status(c).lower()
        status = "metastatic" if c.lower().startswith("metastatic ") else (
                 "core" if strip_status(c).lower() == c.lower() else "other")
        if status == "other":
            continue
        key = (core, status)
        if key not in core_seen:
            core_seen.add(key)
            result.append(c)
    return sorted(set(result))


# ── 2. Aggressive dedup ───────────────────────────────────────────────────────
# Strip AJCC version entirely, strip all status prefixes, deduplicate

def aggressive_dedup(concepts):
    seen = set()
    result = []
    for c in concepts:
        normalised = strip_ajcc(strip_status(c)).strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            result.append(normalised)
    return sorted(result)


mod = moderate_dedup(concepts)
agg = aggressive_dedup(concepts)

out_mod = Path(__file__).parent / "all_concepts_moderate_dedup.txt"
out_agg = Path(__file__).parent / "all_concepts_aggressive_dedup.txt"

out_mod.write_text("\n".join(mod) + "\n")
out_agg.write_text("\n".join(agg) + "\n")
print(f"Moderate dedup:   {len(mod):>6} concepts → {out_mod.name}")
print(f"Aggressive dedup: {len(agg):>6} concepts → {out_agg.name}")


# ── 3. Curated set ────────────────────────────────────────────────────────────

CURATED = [
    # ── Cancer types: TCGA cohort-specific ───────────────────────────────────
    # BRCA
    "invasive ductal carcinoma",
    "invasive lobular carcinoma",
    "invasive breast carcinoma",
    "breast ductal carcinoma in situ",
    "triple negative breast cancer",
    "HER2-positive breast cancer",
    "hormone receptor positive breast cancer",
    # COAD / READ
    "colon adenocarcinoma",
    "colorectal adenocarcinoma",
    "rectal adenocarcinoma",
    "mucinous adenocarcinoma of the colon",
    "microsatellite instability high colorectal cancer",
    # HNSC
    "head and neck squamous cell carcinoma",
    "oral squamous cell carcinoma",
    "oropharyngeal squamous cell carcinoma",
    "laryngeal squamous cell carcinoma",
    "HPV-positive oropharyngeal carcinoma",
    "HPV-negative head and neck squamous cell carcinoma",
    # KICH
    "chromophobe renal cell carcinoma",
    "renal oncocytoma",
    # KIRC
    "clear cell renal cell carcinoma",
    "conventional renal cell carcinoma",
    # KIRP
    "papillary renal cell carcinoma",
    "type 1 papillary renal cell carcinoma",
    "type 2 papillary renal cell carcinoma",
    # LUAD
    "lung adenocarcinoma",
    "pulmonary adenocarcinoma",
    "non-small cell lung cancer",
    "lung adenocarcinoma with lepidic pattern",
    "lung adenocarcinoma with acinar pattern",
    # LUSC
    "lung squamous cell carcinoma",
    "pulmonary squamous cell carcinoma",
    # PAAD
    "pancreatic ductal adenocarcinoma",
    "pancreatic adenocarcinoma",
    "pancreatic intraepithelial neoplasia",
    # General carcinoma types
    "adenocarcinoma",
    "squamous cell carcinoma",
    "urothelial carcinoma",
    "renal cell carcinoma",

    # ── Staging: morphological correlates ────────────────────────────────────
    # Stage descriptors
    "stage I cancer",
    "stage II cancer",
    "stage III cancer",
    "stage IV cancer",
    "early stage cancer",
    "late stage cancer",
    "locally advanced cancer",
    "metastatic cancer",
    # Invasion
    "lymphovascular invasion",
    "perineural invasion",
    "vascular invasion",
    "capsular invasion",
    "extracapsular extension",
    "serosal invasion",
    "pleural invasion",
    "deep tissue invasion",
    # Margins
    "positive surgical margin",
    "negative surgical margin",
    "close surgical margin",
    # Lymph nodes
    "lymph node metastasis",
    "no lymph node metastasis",
    "regional lymph node involvement",
    "extranodal extension",
    "sentinel lymph node positive",
    "sentinel lymph node negative",
    "micrometastasis in lymph node",
    # Distant metastasis
    "distant metastasis",
    "hepatic metastasis",
    "pulmonary metastasis",
    "bone metastasis",

    # ── Grade and differentiation ─────────────────────────────────────────────
    "well differentiated carcinoma",
    "moderately differentiated carcinoma",
    "poorly differentiated carcinoma",
    "undifferentiated carcinoma",
    "grade 1 tumour",
    "grade 2 tumour",
    "grade 3 tumour",
    "high grade carcinoma",
    "low grade carcinoma",
    "high nuclear grade",
    "low nuclear grade",

    # ── Proliferation and necrosis ────────────────────────────────────────────
    "high mitotic rate",
    "low mitotic rate",
    "atypical mitotic figures",
    "high ki67 proliferation index",
    "low ki67 proliferation index",
    "extensive tumour necrosis",
    "focal tumour necrosis",
    "comedonecrosis",

    # ── Growth pattern and architecture ──────────────────────────────────────
    "invasive carcinoma",
    "carcinoma in situ",
    "microinvasive carcinoma",
    "glandular growth pattern",
    "solid growth pattern",
    "papillary growth pattern",
    "micropapillary growth pattern",
    "cribriform growth pattern",
    "infiltrating growth pattern",
    "pushing border",
    "infiltrating border",
    "desmoplastic stroma",

    # ── Immune and stromal features ───────────────────────────────────────────
    "tumour infiltrating lymphocytes",
    "brisk tumour infiltrating lymphocytes",
    "sparse tumour infiltrating lymphocytes",
    "peritumoral inflammation",
    "stromal fibrosis",
    "dense fibrotic stroma",

    # ── Histological staining patterns ───────────────────────────────────────
    "hematoxylin and eosin staining",
    "H&E stained tissue section",
    "nuclear pleomorphism",
    "prominent nucleoli",
    "clear cytoplasm",
    "eosinophilic cytoplasm",
    "mucin production",
    "signet ring cells",
    "keratinisation",
    "intercellular bridges",

    # ── Age / biological markers ──────────────────────────────────────────────
    "young patient tumour",
    "elderly patient tumour",
    "premenopausal breast cancer",
    "postmenopausal breast cancer",
    "high RNA expression tumour",
    "low RNA expression tumour",
    "immune hot tumour",
    "immune cold tumour",
    "microsatellite instability",
    "chromosomal instability",
]

CURATED = sorted(set(CURATED))

out_cur = Path(__file__).parent / "curated_concepts.txt"
out_cur.write_text("\n".join(CURATED) + "\n")
print(f"Curated set:      {len(CURATED):>6} concepts → {out_cur.name}")
