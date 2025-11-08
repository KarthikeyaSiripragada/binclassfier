"""
Smart Bin – CPU Mode Inference (ViT-B/32)
Grid crops only • Stable embeddings
"""

import os, numpy as np, pickle, torch, open_clip
from typing import Dict, List, Tuple
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

BASE = "/content/drive/MyDrive/abid-mvp"
RAW_IMG_DIR = f"{BASE}/data/raw/bin-images"
MODEL_DIR   = f"{BASE}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CLIP_MODEL = None
_CLIP_PREP  = None


def load_clip(model_name="ViT-B-32", pretrained="openai"):
    global _CLIP_MODEL, _CLIP_PREP
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _, _CLIP_PREP = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _CLIP_MODEL = _CLIP_MODEL.to(_DEVICE).eval()
    return _CLIP_MODEL, _CLIP_PREP


def _embed_pil(pil_img):
    model, prep = load_clip()
    with torch.no_grad():
        t = prep(pil_img).unsqueeze(0).to(_DEVICE)
        e = model.encode_image(t)
    return e.cpu().numpy()


def grid_crops(img, grid=3, min_side=64):
    W,H = img.width, img.height
    xs = np.linspace(0,W,grid+1,dtype=int)
    ys = np.linspace(0,H,grid+1,dtype=int)
    crops, boxes = [], []
    for i in range(grid):
        for j in range(grid):
            x1,x2 = xs[i],xs[i+1]
            y1,y2 = ys[j],ys[j+1]
            if x2-x1<min_side or y2-y1<min_side: continue
            crops.append(img.crop((x1,y1,x2,y2)))
            boxes.append([x1,y1,x2,y2])
    return crops, np.array(boxes,dtype=int)


def count_asin_in_image(
    img_path:str, asin:str, asin_emb:Dict[str,np.ndarray],
    score_thresh:float=0.20
) -> Tuple[int,List[Tuple[int,float]],np.ndarray]:

    if asin not in asin_emb:
        return 0,[],np.zeros((0,4),dtype=int)

    img = Image.open(img_path).convert("RGB")
    crops,boxes = grid_crops(img)
    if not crops:
        return 0,[],boxes

    crop_embs = np.vstack([_embed_pil(c) for c in crops])
    sims = cosine_similarity(crop_embs, asin_emb[asin]).ravel()
    keep = np.flatnonzero(sims >= score_thresh)
    hits = sorted(((int(i),float(sims[i])) for i in keep),
                  key=lambda x:(-x[1],x[0]))
    return int(len(keep)), hits, boxes
