# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is a fork of CLIP-Dissect adapted for **computational pathology**: replacing OpenAI CLIP with CONCH (a histopathology VLM), using TCGA tiles as `D_probe`, and pathology concept sets from clinical ontologies (OMOP/PathologyOntology). The target models being dissected are pathology-specific encoders (e.g., CONCH's own image encoder, UNI, CTransPath, or ResNet50 pretrained on pathology data).

## Running the pipeline

```bash
# Default run (ResNet-50 / ImageNet / Broden / CLIP ViT-B16)
python describe_neurons.py

# Custom run
python describe_neurons.py \
  --clip_model ViT-B/16 \
  --target_model resnet50 \
  --target_layers layer3,layer4 \
  --d_probe imagenet_val \
  --concept_set data/20k.txt \
  --batch_size 200 \
  --device cuda \
  --pool_mode avg \
  --similarity_fn soft_wpmi

# Download Broden dataset
bash dlbroden.sh
```

Results are saved to `results/{target_model}_{timestamp}/descriptions.csv`.

## Architecture Overview

CLIP-Dissect has three decoupled components that interact only through cached `.pt` files:

### 1. Target model activation extraction (`utils.py: save_target_activations`)
PyTorch forward hooks are registered on named layers of the **target model** (the DNN being dissected). The dataset is batched through and activations are pooled spatially (avg/max for CNNs, [CLS] token for ViT) to produce `[N_images, N_channels]` tensors saved to disk.

### 2. CLIP embedding extraction (`utils.py: save_clip_image_features`, `save_clip_text_features`)
CLIP encodes all probe images (`model.encode_image`) and all concept strings (`model.encode_text` after tokenization). These produce `[N_images, embed_dim]` and `[N_concepts, embed_dim]` tensors, both L2-normalized, saved to disk.

### 3. Similarity computation (`utils.py: get_similarity_from_activations` + `similarity.py`)
The concept-activation matrix `P = image_features @ text_features.T` (shape `[N_images, N_concepts]`) is computed from cached CLIP embeddings. Each neuron's activation vector over probe images is compared to the columns of `P` using a similarity function. The result is `[N_neurons, N_concepts]`; argmax gives each neuron's description.

The **key insight**: CLIP and the target model are completely decoupled. CLIP only sees probe images (not the target model), and the target model only sees probe images (not CLIP). They are connected only through the shared probe dataset ordering.

### File naming convention for cache
`saved_activations/{d_probe}_{model_name}_{layer}_{pool_mode}.pt`

## Swapping CLIP for CONCH

CONCH's contrastive encoder is a drop-in replacement for CLIP. The integration points are:

| Location | What to change |
|---|---|
| `utils.py: save_activations` | Replace `clip.load(clip_name, device)` with CONCH model loading |
| `utils.py: save_clip_image_features` | Replace `model.encode_image(images)` with CONCH image encoder call |
| `utils.py: save_clip_text_features` / `get_clip_text_features` | Replace `clip.tokenize(text)` + `model.encode_text(tokens)` with CONCH tokenizer + text encoder |
| `data_utils.py: get_data` | Add TCGA tile dataset; CONCH uses 448×448 preprocessing |
| `describe_neurons.py` | Add `--clip_model conch` argument branch |

CONCH's contrastive pooler outputs a single `[batch, 512]` image embedding (same interface as CLIP). The text encoder outputs `[batch, 512]` embeddings at the `<CLS>` token. Both are L2-normalized for cosine similarity — exactly what `get_similarity_from_activations` expects.

**CONCH preprocessing**: images are resized/center-cropped to 448×448, not 224×224. Use `transforms` from the `conch` package (`get_eval_transforms`).

## Adding TCGA as D_probe

Add a new entry to `data_utils.DATASET_ROOTS` and a new branch in `get_data`:
```python
DATASET_ROOTS["tcga"] = "/path/to/tcga_tiles/"  # ImageFolder structure or flat dir
```
TCGA tiles should be organized as `ImageFolder` (subfolders per slide/cancer type) or use a custom `Dataset`. The dataset just needs to be iterable and return `(image_tensor, label)` — labels are unused in CLIP-Dissect.

## Adding pathology concept sets

Concept sets are plain text files (one concept per line). Create `data/pathology_concepts.txt` from OMOP or other ontologies. No other changes needed — `describe_neurons.py` reads them with `--concept_set data/pathology_concepts.txt`.

For pathology, use sentence-style concepts matching CONCH's training distribution (e.g., `"invasive ductal carcinoma"`, `"nuclear pleomorphism"`) rather than single words, since CONCH was trained on captions, not keywords.

## Key dependencies

- `torch`, `torchvision >= 0.13`
- `ftfy`, `regex` — required for CLIP's BPE tokenizer (not needed if fully replacing CLIP)
- `pandas`, `scipy`, `tqdm`
- `conch` package (install from Mahmood Lab GitHub) for the CONCH adaptation
