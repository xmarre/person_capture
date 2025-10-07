import numpy as np
import torch
import cv2
from PIL import Image
import open_clip


class ReIDEmbedder:
    """
    Body embedding via OpenCLIP. Returns L2-normalized embeddings as np.float32.
    """

    def __init__(self, device: str = 'cuda',
                 model_name: str = 'ViT-B-32',
                 pretrained: str = 'laion2b_s34b_b79k'):
        use_cuda = device == 'cuda' and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, img_bgr_list):
        """
        img_bgr_list: list of BGR numpy arrays (H,W,3), uint8.
        returns: list[np.ndarray] of shape (D,), dtype float32.
        """
        if not img_bgr_list:
            return []

        tensors = []
        for bgr in img_bgr_list:
            if bgr is None or bgr.size == 0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil))

        if not tensors:
            return []

        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        feats = self.model.encode_image(batch)
        feats = torch.nn.functional.normalize(feats, dim=1)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        return [f for f in feats]
