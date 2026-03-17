"""
Generate UNI tile embeddings from TCGA tiles.
Saves per-slide .pt files of shape [N_tiles, 1024] to --output_dir.
Tiles are processed in sorted filename order to ensure alignment with PLIP embeddings.
"""
import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm

parser = argparse.ArgumentParser()
parser.add_argument("--tile_dir", type=str, default="/home/maracuja/data/tcga/tiles")
parser.add_argument("--output_dir", type=str, default="/home/maracuja/data/tcga/uni_embeddings")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# UNI preprocessing (224x224, ImageNet normalisation)
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
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


# Load UNI from HuggingFace
print("Loading UNI model...")
model = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
)
model.eval().to(args.device)

tile_dir = Path(args.tile_dir)
slide_dirs = sorted(tile_dir.iterdir())

for slide_dir in tqdm(slide_dirs, desc="Slides"):
    if not slide_dir.is_dir():
        continue

    out_path = Path(args.output_dir) / f"{slide_dir.name}.pt"
    if out_path.exists():
        continue

    # Sorted order is critical for alignment with PLIP
    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        continue

    dataset = SlideTileDataset(tile_paths, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model(batch.to(args.device)))

    torch.save(torch.cat(embeddings).cpu(), out_path)

print(f"Done. Embeddings saved to {args.output_dir}")
