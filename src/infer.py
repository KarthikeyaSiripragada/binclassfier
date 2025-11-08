# src/infer.py
"""
Smart Bin — Inference (robust)
ViT-B/32 CLIP • YOLO-nano proposals • Grid fallback • NMS + fusion
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# heavy/optional libs guarded below
_IMPORT_ERROR: Optional[BaseException] = None
try:
    import torch
    import open_clip
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover - import guard
    _IMPORT_ERROR = e

# Exports
__all__ = [
    "ensure_models_ready",
    "count_asin_in_image",
    "count_multiple_asins",
    "load_clip",
    "load_yolo",
]

# -------------------------
# Paths & base defaults
# -------------------------
# Base detection: prefer SMARTBIN_BASE environment or repo layout
def _find_base() -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(50):
        if (cur / "src" / "infer.py").exists() or (cur / "infer.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd()

BASE = str(Path(os.environ.get("SMARTBIN_BASE") or _find_base()))
RAW_IMG_DIR = os.path.join(BASE, "data", "raw", "bin-images")
PROC_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Globals
# -------------------------
# Use torch.device if torch is available; else fallback to 'cpu' string
_DEVICE = None
if _IMPORT_ERROR is None:
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    _DEVICE = "cpu"  # placeholder, safe for imports

_CLIP_MODEL = None
_CLIP_PREP = None
_YOLO = None

# -------------------------
# Helpers: dependency guard
# -------------------------
def _ensure_deps():
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Missing heavy libs for infer.py. "
            "Install torch, open_clip, ultralytics. Orig: " + repr(_IMPORT_ERROR)
        )

# -------------------------
# Load CLIP
# -------------------------
def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """Lazy-load CLIP once (CPU/GPU safe)."""
    global _CLIP_MODEL, _CLIP_PREP
    _ensure_deps()
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _, _CLIP_PREP = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # move model to torch.device
        _CLIP_MODEL = _CLIP_MODEL.to(_DEVICE).eval()
    return _CLIP_MODEL, _CLIP_PREP

# -------------------------
# Load YOLO
# -------------------------
def load_yolo(weights: str = "yolov8n.pt"):
    """Lazy-load YOLO model (ultralytics)."""
    global _YOLO
    _ensure_deps()
    if _YOLO is None:
        # ultralytics YOLO(...) returns a model wrapper; this is correct usage.
        _YOLO = YOLO(weights)
    return _YOLO

# -------------------------
# Embed PIL image with CLIP
# -------------------------
def _embed_pil(img: Image.Image) -> np.ndarray:
    """
    Embed a PIL image using CLIP transforms -> returns (1, D) numpy array.
    """
    model, prep = load_clip()
    # prep is a torchvision transform that returns a tensor
    with torch.no_grad():
        t = prep(img).unsqueeze(0).to(_DEVICE)  # shape (1,C,H,W)
        e = model.encode_image(t)               # torch tensor (1, D)
    return e.cpu().numpy()

# -------------------------
# Grid fallback
# -------------------------
def grid_crops(img: Image.Image, grid: int = 4, min_side: int = 48):
    W, H = img.width, img.height
    xs = np.linspace(0, W, grid + 1, dtype=int)
    ys = np.linspace(0, H, grid + 1, dtype=int)
    crops: List[Image.Image] = []
    boxes: List[List[int]] = []
    for i in range(grid):
        for j in range(grid):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[j], ys[j + 1]
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            boxes.append([x1, y1, x2, y2])
    return crops, np.array(boxes, dtype=int)

# -------------------------
# IOU + simple NMS hit filter
# -------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

def _nms_hits(hits: List[Tuple[int, float]], boxes: np.ndarray, iou_thr: float = 0.6):
    kept: List[Tuple[int, float]] = []
    removed = set()
    for i, s in hits:
        if i in removed:
            continue
        kept.append((i, s))
        for j, _ in hits:
            if j == i or j in removed:
                continue
            if _iou(boxes[i], boxes[j]) >= iou_thr:
                removed.add(j)
    return kept

# -------------------------
# Main: count asin in image
# -------------------------
def count_asin_in_image(
    img_path: str,
    asin: str,
    asin_emb: Dict[str, np.ndarray],
    score_thresh: float = 0.26,
    yolo_conf: float = 0.003,
    imgsz: int = 320,
    grid: int = 4,
    min_side: int = 48,
    max_crops: int = 16,
    iou_thr: float = 0.6,
) -> Tuple[int, List[Tuple[int, float]], np.ndarray]:
    """
    Returns number of detected hits, list of (crop_idx, score) after NMS sorted desc,
    and boxes array (N,4) corresponding to proposals used.
    """
    # quick checks
    if asin not in asin_emb:
        return 0, [], np.zeros((0, 4), dtype=int)

    # load image
    img = Image.open(img_path).convert("RGB")

    # YOLO proposals
    try:
        yolo = load_yolo()
        res = yolo.predict(source=np.array(img), verbose=False, conf=yolo_conf, imgsz=imgsz)[0]
    except Exception:
        # If YOLO failed for some reason, fallback to empty result
        res = None

    boxes: np.ndarray = np.zeros((0, 4), dtype=int)
    crops: List[Image.Image] = []

    if res is not None and getattr(res, "boxes", None) is not None:
        try:
            # res.boxes.xyxy may be a tensor on cpu
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            boxes = xyxy if xyxy.size else np.zeros((0, 4), dtype=int)
        except Exception:
            boxes = np.zeros((0, 4), dtype=int)

        # extract crops from boxes (clamp)
        W, H = img.width, img.height
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))

    # fallback to grid if YOLO produced nothing usable
    if not crops:
        crops, boxes = grid_crops(img, grid=grid, min_side=min_side)

    # clamp crops count
    if len(crops) > max_crops:
        step = max(1, len(crops) // max_crops)
        keep_idx = list(range(0, len(crops), step))[:max_crops]
        crops = [crops[i] for i in keep_idx]
        if len(boxes):
            boxes = boxes[keep_idx]

    if not crops:
        return 0, [], boxes

    # embed crops
    crop_embs = np.vstack([_embed_pil(c) for c in crops])              # (N, D)
    crop_embs = crop_embs / np.clip(np.linalg.norm(crop_embs, axis=1, keepdims=True), 1e-9, None)

    # get target embedding and normalize
    target_emb = asin_emb[asin]
    # Accept both (D,) or (1,D) or (N,D)
    if target_emb.ndim == 1:
        target_emb = target_emb.reshape(1, -1)
    target_emb = target_emb / np.clip(np.linalg.norm(target_emb, axis=1, keepdims=True), 1e-9, None)

    # cosine similarities
    sims = (crop_embs @ target_emb.T).ravel()

    # optionally fuse YOLO confidence (if available)
    yolo_confs = None
    try:
        if res is not None and getattr(res, "boxes", None) is not None and hasattr(res.boxes, "conf"):
            confs = res.boxes.conf.cpu().numpy().ravel()
            if confs.size >= sims.size:
                yolo_confs = confs[: len(sims)]
    except Exception:
        yolo_confs = None

    if yolo_confs is not None:
        sims = 0.7 * sims + 0.3 * yolo_confs

    # keep indices above threshold
    keep = np.flatnonzero(sims >= score_thresh)
    if len(keep) > max(1, (grid**2) // 3):
        keep = keep[: (grid**2) // 3]

    hits = sorted(((int(i), float(sims[i])) for i in keep), key=lambda x: (-x[1], x[0]))

    # dynamic iou threshold
    dynamic_iou = min(0.8, 0.4 + 0.5 * float(score_thresh))

    # NMS
    hits = _nms_hits(hits, boxes, iou_thr=dynamic_iou)

    # merge near-duplicates
    if len(hits) > 1:
        hits = sorted(hits, key=lambda x: -x[1])
        final_hits, removed = [], set()
        for i, s in hits:
            if i in removed:
                continue
            final_hits.append((i, s))
            for j, sj in hits:
                if j == i or j in removed:
                    continue
                if _iou(boxes[i], boxes[j]) >= 0.45 and abs(s - sj) < 0.10:
                    removed.add(j)
        hits = final_hits

    # drop low-sim tail when many
    if len(hits) > 2:
        sim_vals = [s for _, s in hits]
        drop_thresh = max(sim_vals) - 0.15
        hits = [(i, s) for i, s in hits if s >= drop_thresh]

    return int(len(hits)), hits, boxes

# -------------------------
# Count multiple ASINs (reuse crops)
# -------------------------
def count_multiple_asins(
    img_path: str,
    asins: List[str],
    asin_emb: Dict[str, np.ndarray],
    score_thresh: float = 0.26,
    yolo_conf: float = 0.003,
    imgsz: int = 320,
    grid: int = 4,
    min_side: int = 48,
    max_crops: int = 16,
    iou_thr: float = 0.6,
) -> Dict[str, int]:
    """
    Count multiple ASINs with one crop embedding pass.
    """
    img = Image.open(img_path).convert("RGB")

    try:
        yolo = load_yolo()
        res = yolo.predict(source=np.array(img), verbose=False, conf=yolo_conf, imgsz=imgsz)[0]
    except Exception:
        res = None

    boxes = np.zeros((0, 4), dtype=int)
    crops = []
    if res is not None and getattr(res, "boxes", None) is not None:
        try:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            boxes = xyxy if xyxy.size else np.zeros((0, 4), dtype=int)
        except Exception:
            boxes = np.zeros((0, 4), dtype=int)

        W, H = img.width, img.height
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))

    if not crops:
        crops, boxes = grid_crops(img, grid=grid, min_side=min_side)

    if len(crops) > max_crops:
        step = max(1, len(crops) // max_crops)
        keep = list(range(0, len(crops), step))[:max_crops]
        crops = [crops[i] for i in keep]
        if len(boxes):
            boxes = boxes[keep]

    if not crops:
        return {a: 0 for a in asins}

    crop_embs = np.vstack([_embed_pil(c) for c in crops])
    out = {}
    for asin in asins:
        if asin not in asin_emb:
            out[asin] = 0
            continue
        targ = asin_emb[asin]
        if targ.ndim == 1:
            targ = targ.reshape(1, -1)
        sims = cosine_similarity(crop_embs, targ).ravel()
        keep_idx = np.flatnonzero(sims >= float(score_thresh))
        hits = sorted(((int(i), float(sims[i])) for i in keep_idx), key=lambda x: (-x[1], x[0]))
        hits = _nms_hits(hits, boxes, iou_thr=iou_thr)
        out[asin] = len(hits)
    return out

# -------------------------
# Utilities
# -------------------------
def ensure_models_ready():
    """Force-load CLIP and YOLO."""
    _ensure_deps()
    load_clip()
    load_yolo()

if __name__ == "__main__":
    try:
        ensure_models_ready()
        print("Infer module ready on:", _DEVICE)
    except Exception as e:
        print("Model init failed:", e)
