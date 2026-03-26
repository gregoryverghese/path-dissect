"""
path_dissect.datasets — dataset loading for probe images and target models.

Submodules
----------
tcga     — SlideTileDataset, SlideEmbeddingDataset, TCGA config constants, get_cem_model
standard — DATASET_ROOTS, get_target_model, get_resnet_imagenet_preprocess,
           get_data, get_places_id_to_broden_label, get_cifar_superclass
"""

from .tcga import (
    SlideTileDataset,
    SlideEmbeddingDataset,
    UNI_EMB_DIR,
    PLIP_EMB_DIR,
    CONCH_EMB_DIR,
    CLIP_EMB_DIR,
    CEM_CHECKPOINT,
    CEM_HPARAMS,
    TCGA_TILE_DIR,
    get_cem_model,
)
from .standard import (
    DATASET_ROOTS,
    get_target_model,
    get_resnet_imagenet_preprocess,
    get_data,
    get_places_id_to_broden_label,
    get_cifar_superclass,
)

__all__ = [
    # tcga
    "SlideTileDataset",
    "SlideEmbeddingDataset",
    "UNI_EMB_DIR",
    "PLIP_EMB_DIR",
    "CONCH_EMB_DIR",
    "CLIP_EMB_DIR",
    "CEM_CHECKPOINT",
    "CEM_HPARAMS",
    "TCGA_TILE_DIR",
    "get_cem_model",
    # standard
    "DATASET_ROOTS",
    "get_target_model",
    "get_resnet_imagenet_preprocess",
    "get_data",
    "get_places_id_to_broden_label",
    "get_cifar_superclass",
]
