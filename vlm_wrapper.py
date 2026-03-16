"""
VLM wrappers that expose a unified encode_image / encode_text interface
so utils.py needs no changes when swapping between CLIP, PLIP, CONCH, etc.
"""
import torch
import torch.nn.functional as F


class VLMWrapper:
    """Base class. Subclasses must implement encode_image and encode_text."""

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, C, H, W] tensor, already preprocessed
        returns: [B, D] L2-normalized embeddings
        """
        raise NotImplementedError

    def encode_text(self, tokens) -> torch.Tensor:
        """
        tokens: whatever tokenize() returns for this VLM
        returns: [B, D] L2-normalized embeddings
        """
        raise NotImplementedError

    def tokenize(self, texts: list[str], device: str = "cpu"):
        """Tokenize a list of strings. Returns input for encode_text."""
        raise NotImplementedError

    @property
    def preprocess(self):
        """torchvision-compatible transform for probe images."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CLIP (OpenAI)
# ---------------------------------------------------------------------------

class CLIPWrapper(VLMWrapper):
    def __init__(self, model_name: str, device: str):
        import clip as openai_clip
        self.model, self._preprocess = openai_clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
        self._clip = openai_clip

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def tokenize(self, texts, device="cpu"):
        return self._clip.tokenize(texts).to(device)

    @property
    def preprocess(self):
        return self._preprocess


# ---------------------------------------------------------------------------
# PLIP (vinid/plip on HuggingFace — same arch as CLIP ViT-B/32)
# ---------------------------------------------------------------------------

class PLIPWrapper(VLMWrapper):
    def __init__(self, device: str):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("vinid/plip").to(device)
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.model.eval()
        self.device = device

    def encode_image(self, images):
        # images: [B, C, H, W] float tensor, already preprocessed
        feats = self.model.get_image_features(pixel_values=images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, tokens):
        # tokens: dict of tensors from tokenize()
        feats = self.model.get_text_features(**tokens)
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


# ---------------------------------------------------------------------------
# CONCH (Mahmood Lab — contrastive pathology VLM, 448×448)
# ---------------------------------------------------------------------------

class CONCHWrapper(VLMWrapper):
    def __init__(self, checkpoint_path: str, device: str):
        """
        checkpoint_path: path to CONCH model weights
        Install conch package from https://github.com/mahmoodlab/CONCH
        """
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        self.model, self._preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path
        )
        self.tokenizer = get_tokenizer()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def encode_image(self, images):
        feats = self.model.encode_image(images, proj_contrast=True, normalize=True)
        return feats

    def encode_text(self, tokens):
        feats = self.model.encode_text(tokens, normalize=True)
        return feats

    def tokenize(self, texts, device="cpu"):
        from conch.open_clip_custom import tokenize as conch_tokenize
        return conch_tokenize(self.tokenizer, texts).to(device)

    @property
    def preprocess(self):
        return self._preprocess


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_vlm(clip_name: str, device: str, **kwargs) -> VLMWrapper:
    """
    clip_name: "plip", "conch", or any OpenAI CLIP model name (e.g. "ViT-B/16")
    kwargs: passed to wrapper constructors (e.g. checkpoint_path for CONCH)
    """
    if clip_name == "plip":
        return PLIPWrapper(device)
    elif clip_name == "conch":
        checkpoint_path = kwargs.get("conch_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("Pass conch_checkpoint=<path> when using CONCH")
        return CONCHWrapper(checkpoint_path, device)
    else:
        return CLIPWrapper(clip_name, device)
