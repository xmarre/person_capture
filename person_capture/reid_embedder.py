from typing import TYPE_CHECKING, List
import numpy as np
import cv2
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import open_clip as open_clip_type

class ReIDEmbedder:
    """Body embedding via OpenCLIP.
    Returns L2-normalized embeddings as np.float32.
    """
    def __init__(self, device: str = 'cuda',
                 clip_model_name: str = 'ViT-L-14',
                 clip_pretrained: str = 'laion2b_s32b_b82k',
                 progress=None):
        try:
            import torch as _torch
            import open_clip as _open_clip
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Heavy dependencies not installed; install requirements.txt to run reid.") from e
        self._torch = _torch
        self.device = 'cuda' if (str(device).startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.model, _, self.preprocess = _open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
        self.model.eval().to(self.device)
        self.progress = progress

    def extract(self, bgr_list: List) -> List[np.ndarray]:
        if not bgr_list:
            return []
        tensors = []
        for bgr in bgr_list:
            if bgr is None or getattr(bgr, 'size', 0) == 0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil))
        if not tensors:
            return []
        batch = self._torch.stack(tensors).to(self.device, non_blocking=True)
        with self._torch.inference_mode():
            feats = self.model.encode_image(batch)
            feats = self._torch.nn.functional.normalize(feats, dim=1)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        return [f for f in feats]
