# app_finished_viewer.py — Viewer for "finished" / eval_outputs images + annotations
# Usage: streamlit run app_finished_viewer.py
import os, io, json, statistics
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
# ---- safe rerun helper (simple, cross-version) ----
# Use st.experimental_rerun if it exists; otherwise show a warning.
if hasattr(st, "experimental_rerun"):
    _st_rerun = st.experimental_rerun # type: ignore
else:
    def _st_rerun():
        # Graceful fallback for Streamlit builds without experimental_rerun.
        # This avoids importing internal APIs and keeps linters happy.
        st.warning("Auto-reload not available in this Streamlit build — please refresh the page to reload.")

# ---- safe rerun helper (cross-version) ----
try:
    _st_rerun = st.experimental_rerun  # type: ignore
except Exception:
    try:
        from streamlit.runtime.scriptrunner import RerunException
        def _st_rerun(): # type: ignore
            # Trigger Streamlit internal rerun
            raise RerunException(None) # type: ignore
    except Exception:
        def _st_rerun():
            # Last-resort: user must manually refresh
            st.warning("Auto-reload not available in this Streamlit build — please refresh the page to reload.")

# ---- CONFIG (adjust if your folder names differ) ----
BASE = Path(__file__).resolve().parent
FINISHED_DIR = BASE / "data" / "finished"
FINISHED_IMG_DIR = FINISHED_DIR / "images"             # preferred
ALT_EVAL_DIR = FINISHED_DIR / "eval_outputs"           # fallback
ANNOT_PATH = FINISHED_DIR / "annotations.json"

# ---- helpers ----
def find_image_dir():
    if FINISHED_IMG_DIR.exists() and any(FINISHED_IMG_DIR.iterdir()):
        return FINISHED_IMG_DIR
    if ALT_EVAL_DIR.exists() and any(ALT_EVAL_DIR.iterdir()):
        return ALT_EVAL_DIR
    return None

def load_annotations(p: Path):
    if not p.exists():
        return []
    try:
        txt = p.read_text(encoding="utf-8")
        return json.loads(txt) if txt.strip() else []
    except Exception:
        return []

def draw_boxes(img: Image.Image, boxes, confs=None):
    draw = ImageDraw.Draw(img)
    for idx, b in enumerate(boxes):
        try:
            x1,y1,x2,y2 = map(int, b)
            draw.rectangle([x1,y1,x2,y2], outline="lime", width=3)
            if confs and idx < len(confs):
                try:
                    draw.text((x1+4, y1+4), f"{float(confs[idx]):.2f}", fill="yellow")
                except Exception:
                    pass
        except Exception:
            continue
    return img

def _ann_score(ann):
    boxes = ann.get("boxes") or ann.get("bboxes") or ann.get("bboxes_xyxy") or ann.get("boxes_xyxy") or []
    confs = ann.get("confs") or ann.get("confs_conf") or ann.get("confs_float") or None
    count = len(boxes)
    if confs:
        try:
            confs_f = [float(x) for x in confs if x is not None]
            if confs_f:
                return count + statistics.mean(confs_f)
        except Exception:
            pass
    return float(count)

# ---- UI ----
st.set_page_config(layout="wide", page_title="Finished Images Viewer")
st.title("📦 Finished / Preprocessed Images — Viewer")

img_dir = find_image_dir()
annotations = load_annotations(ANNOT_PATH)

if not img_dir:
    st.warning("No finished images folder found under `data/finished/`. Place preprocessed images in `data/finished/images/` or `data/finished/eval_outputs/`.")
    st.stop()

if not annotations:
    st.info("No annotations.json found — images will still be listed but no boxes will be drawn.")

# Build index: map filename -> list of annotation entries
index = {}
for a in annotations:
    fn = a.get("image") or a.get("file") or a.get("filename") or a.get("file_name")
    if not fn:
        continue
    index.setdefault(fn, []).append(a)

# ---------- Prepare image list ----------
# build clean list of names (strings only), drop any empty names
all_images = sorted([
    str(p.name)
    for p in img_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name.strip()
])

if not all_images:
    st.warning(f"No image files found in `{img_dir}`.")
    st.stop()

# DEBUG toggles exact repr output (set to False for production)
DEBUG = False
if DEBUG:
    st.text("Sample files (repr):")
    for n in all_images[:20]:
        st.write(repr(n))

col1, col2 = st.columns([1,3])

with col1:
    st.subheader("Browse")
    asin_filter = st.text_input("Filter by ASIN (optional)")

    def _fmt(name):
        return name if len(name) <= 48 else name[:45] + "..."

    chosen = st.selectbox("Choose image", options=all_images, format_func=_fmt, index=0)

with col2:
    if not chosen:
        st.info("Pick an image on the left to preview the finished overlay + annotation details.")
    else:
        img_path = img_dir / chosen
        st.markdown(f"**File:** `{chosen}`")
        ann_list = index.get(chosen, [])

        # filter if ASIN search is used
        if asin_filter:
            ann_list = [a for a in ann_list if str(a.get("asin", "")).lower() == asin_filter.lower()]

        if not ann_list:
            boxes, confs = [], None
            st.markdown("No annotation entry for this image (displaying raw image).")
        else:
            # Always pick the best annotation automatically
            scores = [_ann_score(a) for a in ann_list]
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            ann = ann_list[best_idx]
            st.caption(f"Best annotation auto-selected (score={scores[best_idx]:.3f}).")

            boxes = ann.get("boxes") or ann.get("bboxes") or ann.get("bboxes_xyxy") or ann.get("boxes_xyxy") or []
            confs = ann.get("confs") or ann.get("confs_conf") or ann.get("confs_float") or None

            asin = ann.get("asin", "N/A")
            count = ann.get("count", len(boxes))
            st.markdown(f"**ASIN:** `{asin}` — **Count:** {count} — **Boxes:** {len(boxes)}")

        try:
            pil = Image.open(img_path).convert("RGB")
            vis = draw_boxes(pil.copy(), boxes, confs=confs)
            st.image(vis, use_container_width=True)

            bio = io.BytesIO()
            vis.save(bio, "JPEG")
            bio.seek(0)
            dl_name = f"overlay_{chosen}" if chosen.lower().endswith(".jpg") else f"overlay_{chosen}.jpg"
            st.download_button("Download overlay (jpg)", data=bio.read(), file_name=dl_name, mime="image/jpeg")
        except Exception as e:
            st.error(f"Could not open or render image: {e}")
st.caption("Project completed by Karthikeya Siripragada (SE22UECM018) and Karthik Raj Gupta (SE22UCAM004)")
