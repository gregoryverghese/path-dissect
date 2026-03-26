"""
path_dissect.datasets.tcga — TCGA-specific datasets, constants, and helpers.

Contents
--------
SlideTileDataset       — loads tiles from a single slide directory for embedding generation
SlideEmbeddingDataset  — loads pre-computed per-slide UNI embeddings for CEM input
UNI_EMB_DIR            — default directory for UNI embeddings
PLIP_EMB_DIR           — default directory for PLIP embeddings
CONCH_EMB_DIR          — default directory for CONCH embeddings
CLIP_EMB_DIR           — default directory for CLIP embeddings
CEM_CHECKPOINT         — path to trained CEM checkpoint
CEM_HPARAMS            — hyperparameters matching the checkpoint state dict
get_cem_model()        — load ConceptEmbeddingModel from a Lightning checkpoint
"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

UNI_EMB_DIR    = "/home/maracuja/data/tcga/uni_embeddings"
PLIP_EMB_DIR   = "/home/maracuja/data/tcga/plip_embeddings"
CONCH_EMB_DIR  = "/home/maracuja/data/tcga/conch_embeddings"
CLIP_EMB_DIR   = "/home/maracuja/data/tcga/clip_embeddings"
CEM_CHECKPOINT = "/home/maracuja/projects/path-dissect/model.ckpt"

# Inferred from checkpoint state dict
CEM_HPARAMS = dict(
    n_concepts=4,
    n_tasks=1,
    h_dim=1024,
    emb_size=8,
    concept_states=[4, 3, 10, 3],
    n_att_heads=4,
    attn_dim=256,
    dropout=0.4,
    attn_dropout=0.4,
    pre_bn_mlp=False,
    task_type="cox",
    c2y_layers=[],
)


# ---------------------------------------------------------------------------
# TCGA tile ImageFolder root (used by datasets.standard.DATASET_ROOTS)
# ---------------------------------------------------------------------------

TCGA_TILE_DIR = "/home/maracuja/data/tcga/tiles/"


# ---------------------------------------------------------------------------
# Tile dataset (used by embedding generation scripts)
# ---------------------------------------------------------------------------

class SlideTileDataset(Dataset):
    """
    Loads all PNG tiles from a single slide directory, applying a transform.
    Tiles are returned in sorted filename order to ensure alignment across VLMs.
    """
    def __init__(self, tile_dir: Path, transform):
        self.tile_paths = sorted(tile_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        img = Image.open(self.tile_paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Slide embedding dataset
# ---------------------------------------------------------------------------

class SlideEmbeddingDataset(Dataset):
    """
    Loads pre-computed per-slide UNI embeddings for CEM input.
    Returns (embeddings [n_tiles, h_dim], slide_id) in sorted order.
    """
    def __init__(self, emb_dir=UNI_EMB_DIR):
        self.emb_paths = sorted(Path(emb_dir).glob("*.pt"))
        self.slide_ids = [p.stem for p in self.emb_paths]

    def __len__(self):
        return len(self.emb_paths)

    def __getitem__(self, idx):
        emb = torch.load(self.emb_paths[idx], map_location="cpu")
        return emb, self.slide_ids[idx]


# ---------------------------------------------------------------------------
# CEM model loader
# ---------------------------------------------------------------------------

class _DummyHook:
    """Dummy attention hook — satisfies the architecture init without recording anything."""
    @staticmethod
    def hook_fn(model, input, output):
        pass


def get_cem_model(checkpoint_path=CEM_CHECKPOINT, device="cuda"):
    """
    Load ConceptEmbeddingModel from a Lightning checkpoint.
    Requires the ccem subpackage (losses.py and attention.py) to be present.
    """
    from ..probe_models.ccem.cem_mil import ConceptEmbeddingModel  # noqa: E402

    model = ConceptEmbeddingModel(**CEM_HPARAMS, attention_hook=_DummyHook())
    ck = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model
