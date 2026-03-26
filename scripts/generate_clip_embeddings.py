"""
Generate CLIP slide embeddings from TCGA tiles.
Tiles are processed in sorted filename order (same as CONCH/PLIP/UNI) to ensure alignment.
Saves per-slide .pt files of shape [1, 512] (mean-pooled, L2-normalised) to --output_dir.
"""
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from path_dissect.vlms.clip import CLIPWrapper
from path_dissect.datasets.tcga import SlideTileDataset, CLIP_EMB_DIR, TCGA_TILE_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--tile_dir",    type=str, default=TCGA_TILE_DIR)
parser.add_argument("--output_dir",  type=str, default=CLIP_EMB_DIR)
parser.add_argument("--clip_model",  type=str, default="ViT-B/16")
parser.add_argument("--image_size",  type=int, default=448,
                    help="Tile resolution. Use 448 to match CONCH preprocessing.")
parser.add_argument("--batch_size",  type=int, default=128)
parser.add_argument("--device",      type=str, default="cuda")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

print(f"Loading CLIP {args.clip_model} (image_size={args.image_size})...")
vlm = CLIPWrapper(args.clip_model, args.device, image_size=args.image_size)

tile_dir = Path(args.tile_dir)
slide_dirs = sorted(d for d in tile_dir.iterdir() if d.is_dir())

for slide_dir in tqdm(slide_dirs, desc="Slides"):
    out_path = Path(args.output_dir) / f"{slide_dir.name}.pt"
    if out_path.exists():
        continue

    dataset = SlideTileDataset(slide_dir, vlm.preprocess)
    if len(dataset) == 0:
        continue

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(vlm.encode_image(batch.to(args.device)))

    # Mean-pool all tile embeddings → single slide embedding [1, 512]
    slide_embedding = torch.cat(embeddings).mean(dim=0, keepdim=True)
    slide_embedding = F.normalize(slide_embedding, dim=-1)
    torch.save(slide_embedding.cpu(), out_path)

print(f"Done. Embeddings saved to {args.output_dir}")
