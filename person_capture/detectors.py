import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    import torch; from ultralytics import YOLO as YOLOType

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', device='cuda', progress=None):
        try:
            import torch as _torch
            # Ensure Ultralytics home before it initializes internal paths
            _pkg_dir = Path(__file__).resolve().parent
            _repo_root = _pkg_dir.parent if _pkg_dir.name == "person_capture" else _pkg_dir
            os.environ.setdefault("ULTRALYTICS_HOME", str(_repo_root / ".ultralytics"))
            Path(os.environ["ULTRALYTICS_HOME"]).mkdir(parents=True, exist_ok=True)
            from ultralytics import YOLO as _YOLO
            # Point Ultralytics HOME into repo root to avoid user roaming caches
            _home = Path(os.environ["ULTRALYTICS_HOME"])
            (_home / "weights").mkdir(parents=True, exist_ok=True)
            (_home / "datasets").mkdir(parents=True, exist_ok=True)
            logging.getLogger(__name__).info("YOLO home=%s", _home)
        except Exception as e:  # pragma: no cover - executed only when deps missing
            raise RuntimeError(
                "Heavy dependencies not installed; install requirements.txt to run detection."
            ) from e

        self._torch = _torch
        self._YOLO = _YOLO
        self.device = 'cuda' if (str(device).startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.progress = progress
        self.model = self._load_model(model_name)
        # Make ORT/TRT engine options stable (avoid rebuilds due to debug flags)
        os.environ.setdefault("ORT_TRT_DUMP_SUBGRAPHS", "0")
        os.environ.setdefault("ORT_TRT_DETAILED_BUILD_LOG", "0")
        try:
            pt_path = Path(getattr(self.model, "pt_path", ""))
            if pt_path.is_file():
                digest = hashlib.sha1(pt_path.read_bytes()[: 1 << 20]).hexdigest()
                self._engine_tag = digest[:8]
            else:
                self._engine_tag = "default"
        except Exception:
            self._engine_tag = "default"
        try:
            # show exactly where weights came from
            m = Path(model_name)
            if m.is_file():
                resolved = m
            else:
                base = m.name.lower()
                hub = base if base.endswith(".pt") else "yolov8n.pt"
                resolved = Path(os.environ["ULTRALYTICS_HOME"]) / "weights" / Path(hub)
            logging.getLogger(__name__).info("YOLO weights=%s", resolved)
        except Exception:
            pass
        try:
            self.model.fuse()
            if self.device == 'cuda':
                self.model.model.half()
        except Exception:
            pass

    def _load_model(self, model_name: str):
        m = Path(model_name)
        # If user gave a real file, load it. Otherwise use hub name to honor ULTRALYTICS_HOME.
        if m.is_file():
            model_arg = str(m)
        else:
            base = m.name.lower()
            hub = base if base.endswith(".pt") else "yolov8n.pt"
            # Always resolve to a concrete local file path and use that for loading
            from ultralytics import settings as yolo_settings

            home = Path(os.environ.get("ULTRALYTICS_HOME", Path.cwd() / ".ultralytics"))
            weights_dir = home / "weights"
            yolo_settings.update({
                "weights_dir": str(weights_dir),
                "datasets_dir": str(home / "datasets"),
            })
            try:
                yolo_settings.save()
            except Exception:
                pass
            local = weights_dir / hub
            log = logging.getLogger(__name__)

            def load_or_quarantine(candidate: Path):
                try:
                    return self._YOLO(str(candidate))
                except Exception as exc:  # pragma: no cover - depends on corrupt cache
                    log.warning("Failed to load YOLO weights at %s (%s)", candidate, exc)
                    if self.progress:
                        self.progress(f"YOLO load failed ({exc}). Recovering...")
                    try:
                        bad = candidate.with_name(candidate.name + ".bad")
                        idx = 1
                        while bad.exists():
                            bad = candidate.with_name(f"{candidate.name}.bad{idx}")
                            idx += 1
                        candidate.replace(bad)
                        log.info("Quarantined corrupt YOLO weights %s → %s", candidate, bad)
                    except Exception:
                        pass
                    return None

            if local.is_file():
                log.info("YOLO weights (cached): %s", local)
                model = load_or_quarantine(local)
                if model is not None:
                    return model

            exists = local.exists()
            size = (local.stat().st_size if exists else 0)
            log.info("YOLO cache probe %s exists=%s size=%d", local, exists, size)
            if exists and size > 0:
                m2 = load_or_quarantine(local)
                if m2 is not None:
                    return m2

            log.info("YOLO cache miss for %s → seeding %s", hub, local)
            try:
                weights_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # Offline-first: prefer an existing file in CWD before any hub attempt
            cwd_candidate = Path.cwd() / hub
            if cwd_candidate.is_file():
                try:
                    if not local.is_file():
                        shutil.copy2(cwd_candidate, local)
                        log.info("Seeded YOLO cache from CWD: %s", cwd_candidate)
                except Exception as e:
                    log.warning("Failed to seed from CWD (%s)", e)
                model = load_or_quarantine(local)
                if model is not None:
                    return model
                # last attempt with the cwd file path (absolute), still not a hub alias
                if cwd_candidate.is_file():
                    m2 = load_or_quarantine(cwd_candidate.resolve())
                    if m2 is not None:
                        return m2

            # Last resort: fetch ONCE, then rebind to the cached absolute path.
            # Using YOLO(hub) only to download; then copy its resolved .pt into our cache.
            try:
                tmp_model = self._YOLO(hub)  # may download to Ultralytics’ own path
                src = Path(getattr(tmp_model, "pt_path", ""))  # Ultralytics resolves this
                if not src.is_file():
                    # fallback: if Ultralytics placed the file in CWD
                    src = Path.cwd() / hub
                if src.is_file():
                    try:
                        weights_dir.mkdir(parents=True, exist_ok=True)
                        if src.resolve() != local.resolve():
                            shutil.copy2(src, local)
                        log.info("Seeded YOLO cache: %s", local)
                        seeded_model = load_or_quarantine(local)
                        if seeded_model is not None:
                            log.info("YOLO loaded from cache: %s", local)
                            return seeded_model
                    except Exception as e:
                        log.warning("Failed to copy YOLO weights into cache (%s)", e)
                else:
                    log.warning("Ultralytics did not expose a concrete pt_path for %s", hub)
            except Exception as e:
                log.warning("YOLO hub fetch failed (%s); continuing without hub", e)

            # Now load strictly from our cached absolute path (no hub alias involved)
            if local.exists() and local.stat().st_size > 0:
                model = load_or_quarantine(local)
                if model is not None:
                    return model
            raise RuntimeError(f"YOLO weights not found (expected at {local})")
        try:
            return self._YOLO(model_arg)
        except Exception as e:
            if self.progress:
                self.progress(f"YOLO load failed ({e}). Recovering...")
            # If user pointed to a local file, quarantine it and try hub name
            try:
                p = m if m.is_file() else Path()
                if p.is_file():
                    bad = p.with_suffix(p.suffix + '.bad')
                    p.rename(bad)
                    if self.progress:
                        self.progress(f"Quarantined corrupt weights: {bad.name}")
            except Exception:
                pass
            # Derive a clean hub model name
            base = Path(model_name).name.lower()
            hub = base if base.endswith('.pt') else 'yolov8n.pt'
            return self._load_model(hub)

    def detect(self, frame, conf=0.35):
        """Return list of dicts for class=person only."""
        try:
            # Ultralytics auto-handles dtype from model; ensure fp16 path on CUDA model
            res = self.model.predict(
                frame,
                device=self.device,
                conf=float(conf),
                iou=0.45,
                classes=[0],
                verbose=False,
                max_det=40,
                imgsz=640,
            )[0]
        except Exception:
            return []

        out = []
        bxs = getattr(res, "boxes", None)
        if bxs is None:
            return out
        for i in range(len(bxs)):
            xyxy = bxs.xyxy[i].tolist()
            c = float(bxs.conf[i].item()) if getattr(bxs, "conf", None) is not None else 0.0
            out.append({"xyxy": xyxy, "conf": c, "cls": 0})
        return out
