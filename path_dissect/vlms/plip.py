"""
PLIPWrapper — wraps vinid/plip (HuggingFace, same arch as CLIP ViT-B/32).
"""
import torch
import torch.nn.functional as F
from .base import VLMWrapper


class PLIPWrapper(VLMWrapper):
    def __init__(self, device: str):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("vinid/plip").to(device)
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.model.eval()
        self.device = device

    def encode_image(self, images):
        # images: [B, C, H, W] float tensor, already preprocessed
        vision_outputs = self.model.vision_model(pixel_values=images)
        feats = self.model.visual_projection(vision_outputs.pooler_output)
        return F.normalize(feats, dim=-1)

    def encode_text(self, tokens):
        # tokens: dict of tensors from tokenize()
        text_outputs = self.model.text_model(**tokens)
        feats = self.model.text_projection(text_outputs.pooler_output)
        return F.normalize(feats, dim=-1)

    def tokenize(self, texts, device="cpu"):
        enc = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        return {k: v.to(device) for k, v in enc.items() if k != "pixel_values"}

    @property
    def preprocess(self):
        from torchvision import transforms
        # PLIP uses the same preprocessing as CLIP ViT-B/32
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])
