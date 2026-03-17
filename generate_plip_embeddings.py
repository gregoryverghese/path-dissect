"""
Generate PLIP slide embeddings from TCGA tiles.
Tiles are processed in sorted filename order (same as UNI) to ensure alignment.
Saves per-slide .pt files of shape [1, 512] (mean-pooled) to --output_dir.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--tile_dir", type=str, default="/home/maracuja/data/tcga/tiles")
parser.add_argument("--output_dir", type=str, default="/home/maracuja/data/tcga/plip_embeddings")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])


class SlideTileDataset(Dataset):
    def __init__(self, tile_paths, transform):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        img = Image.open(self.tile_paths[idx]).convert("RGB")
        return self.transform(img)


print("Loading PLIP model...")
model = CLIPModel.from_pretrained("vinid/plip").to(args.device)
model.eval()

tile_dir = Path(args.tile_dir)
slide_dirs = sorted(tile_dir.iterdir())

for slide_dir in tqdm(slide_dirs, desc="Slides"):
    if not slide_dir.is_dir():
        continue

    out_path = Path(args.output_dir) / f"{slide_dir.name}.pt"
    if out_path.exists():
        continue

    # Sorted order must match UNI embedding generation
    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        continue

    dataset = SlideTileDataset(tile_paths, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            vision_outputs = model.vision_model(pixel_values=batch.to(args.device))
            feats = model.visual_projection(vision_outputs.pooler_output)
            embeddings.append(F.normalize(feats, dim=-1))

    # Mean-pool all tile embeddings → single slide embedding [1, 512]
    slide_embedding = torch.cat(embeddings).mean(dim=0, keepdim=True)
    torch.save(slide_embedding.cpu(), out_path)

print(f"Done. Embeddings saved to {args.output_dir}")
