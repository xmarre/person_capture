
import numpy as np
import torch
import cv2
from PIL import Image
import open_clip

class ReIDEmbedder:
    """
    Body embedding via OpenCLIP.
    Defaults to ViT-L-14 for stronger separation; configurable.
    Returns L2-normalized embeddings as np.float32.
    """

    def __init__(self, device: str = 'cuda',
                 model_name: str = 'ViT-L-14',
                 pretrained: str = 'laion2b_s32b_b82k',
                 progress=None):
        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        if False and progress: progress(f"Preparing ReID OpenCLIP {model_name} {pretrained} (will download if missing)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        pass
        self.model.eval().to(self.device)

    def extract(self, bgr_list):
        if not bgr_list:
            return []
        tensors = []
        for bgr in bgr_list:
            if bgr is None or bgr.size == 0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil))

        if not tensors:
            return []

        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        with torch.inference_mode():
            feats = self.model.encode_image(batch)
            feats = torch.nn.functional.normalize(feats, dim=1)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        return [f for f in feats]
