---
name: CONCH-Dissect project goal and current state
description: Full state of the CONCH-Dissect project — what's done, what's next
type: project
---

Adapting CLIP-Dissect for computational pathology. The target model being probed is a CBM trained on TCGA survival prediction.

## Architecture

- **VLM (reference model)**: CONCH — replaces CLIP for computing concept-activation matrix P
- **Target model**: CBM trained for survival prediction
  - UNI encoder (frozen, pretrained at 224×224) → pre-computed embeddings
  - Attention heads → Concept bottleneck layer (neurons of interest) → Linear/MLP survival head
  - Saved as a .pt checkpoint
- **D_probe**: TCGA slides tiled at **448×448 at 20x magnification** (being generated now)
- **Concept sets**: Three files in data/
  - `data/pathology_concepts.txt` — 12,211 concepts from NCIt + HPO
  - `data/tcga_concepts.txt` — 2,321 concepts from TCGA diagnostic reports
  - `data/pathology_concepts_combined.txt` — 14,467 combined (recommended)

## Pipeline flow

1. CONCH processes raw 448×448 tiles → image embeddings + text concept embeddings → matrix P [N_tiles, N_concepts]
2. UNI processes same tiles **downsampled to 224×224** → pre-computed embeddings (Option A chosen)
3. CBM takes UNI embeddings (in same tile order as CONCH) → hooks into concept bottleneck layer → [N_tiles, N_bottleneck_neurons]
4. CLIP-Dissect similarity functions match each bottleneck neuron to best concept in P

**Critical requirement**: tile ordering must be identical between CONCH and UNI→CBM sides.

## What's done

- CLAUDE.md written
- build_concept_set.py — fetches NCIt + HPO concepts (run and saved)
- build_tcga_concepts.py — extracts concepts from TCGA_Reports.csv (run and saved)
- All three concept files generated in data/
- Tiling decision made: 448×448 at 20x

## What's next (pick up here)

1. User is currently generating TCGA tiles at 448×448 20x
2. Recompute UNI embeddings at 224×224 from the new tiles (Option A)
3. Adapt the codebase:
   - `utils.py`: replace CLIP with CONCH (encode_image, encode_text, tokenizer)
   - `data_utils.py`: add TCGA tile dataset + CONCH 448×448 preprocessing + UNI 224×224 preprocessing
   - `describe_neurons.py`: add conch as clip_model option; handle UNI embeddings as input to target model
   - Need to see CBM model class definition to know layer names for hooking
4. Need CBM checkpoint path and model class to wire up target model loading

**Why Option A**: UNI trained at 224×224 (ViT-Large, 16×16 patches). CONCH trained at 448×448. Both get their native resolution; tiles downsampled 2x for UNI.
