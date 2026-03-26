"""
CLIPWrapper — wraps OpenAI CLIP models.
"""
import torch.nn.functional as F
from torchvision import transforms
from .base import VLMWrapper

# CLIP normalisation constants (same as PLIP — both use OpenAI CLIP preprocessing)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class CLIPWrapper(VLMWrapper):
    def __init__(self, model_name: str, device: str, image_size: int | None = None):
        """
        model_name:  OpenAI CLIP model name, e.g. "ViT-B/16"
        image_size:  Override the default CLIP preprocessing resolution.
                     Use 448 when tiles were generated at that resolution.
                     Defaults to the model's native size (224 for ViT-B/16).
        """
        import clip as openai_clip
        self.model, default_preprocess = openai_clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
        self._clip = openai_clip

        if image_size is not None:
            self._preprocess = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ])
        else:
            self._preprocess = default_preprocess

    def encode_image(self, images):
        feats = self.model.encode_image(images)
        return F.normalize(feats.float(), dim=-1)

    def encode_text(self, tokens):
        feats = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    def tokenize(self, texts, device="cpu"):
        return self._clip.tokenize(texts).to(device)

    @property
    def preprocess(self):
        return self._preprocess
