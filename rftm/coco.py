"""COCO val2017 subset and YOLO -> COCO-format detections, as in notebooks 05 / 05b."""
import contextlib
import glob
import io
import json
import os
import random
import subprocess

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET_PATH = os.path.join(HERE, "..", "prior", "05", "subset_ids.json")


def download_coco(coco_dir):
    """val2017 images (~0.8 GB) and annotations (~250 MB), each only if missing (an interrupted run can leave one)."""
    os.makedirs(coco_dir, exist_ok=True)
    sh = lambda cmd: subprocess.run(cmd, shell=True, check=True)
    if len(glob.glob(f"{coco_dir}/val2017/*.jpg")) < 5000:
        sh(f"rm -rf {coco_dir}/val2017 {coco_dir}/val2017.zip")
        sh(f"wget -q http://images.cocodataset.org/zips/val2017.zip -O {coco_dir}/val2017.zip")
        sh(f"unzip -q -o {coco_dir}/val2017.zip -d {coco_dir} && rm {coco_dir}/val2017.zip")
    if not os.path.exists(f"{coco_dir}/annotations/instances_val2017.json"):
        sh(f"wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O {coco_dir}/ann.zip")
        sh(f"unzip -q -o {coco_dir}/ann.zip -d {coco_dir} && rm {coco_dir}/ann.zip")
    n = len(glob.glob(f"{coco_dir}/val2017/*.jpg"))
    assert n == 5000, f"{n} COCO images on disk, want 5000: is Settings -> Internet ON? Re-run this cell"


def load_coco_gt(coco_dir):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(f"{coco_dir}/annotations/instances_val2017.json")


def subset_ids(coco_gt, n=300, seed=0):
    """Notebook 05's subset: shuffle all val images with >= 1 non-crowd object with Random(seed), take n, sort.
    Checked against the frozen list in prior/05/subset_ids.json."""
    ids = sorted(i for i in coco_gt.getImgIds() if len(coco_gt.getAnnIds(imgIds=i, iscrowd=False)) > 0)
    random.Random(seed).shuffle(ids)
    ids = sorted(ids[:n])
    frozen = json.load(open(SUBSET_PATH))
    assert ids == frozen, "regenerated subset differs from notebook 05's: not the same setup"
    return ids


def load_coco_rgb(coco_gt, coco_dir, image_id):
    info = coco_gt.loadImgs(image_id)[0]
    return cv2.imread(f"{coco_dir}/val2017/{info['file_name']}", cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0


def to_uint8_bgr(img):
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)[:, :, ::-1]   # YOLO reads numpy arrays as BGR


def yolo_to_coco_ids(detector, coco_gt):
    """YOLO's class index i -> the i-th smallest COCO category id (YOLO's 80 classes are COCO's, in id order)."""
    cat_ids = sorted(coco_gt.getCatIds())
    assert len(cat_ids) == 80
    names = [coco_gt.loadCats(c)[0]["name"] for c in cat_ids]
    bad = [(i, detector.names[i], names[i]) for i in range(80) if detector.names[i] != names[i]]
    assert not bad, f"YOLO / COCO class names disagree: {bad[:5]}"
    return cat_ids


def detect(detector, cat_ids, img_rgb_float, image_id, device):
    """One float RGB image -> COCO result dicts. conf=0.001, iou=0.7, max_det=300, exactly as notebook 05."""
    r = detector.predict(np.ascontiguousarray(to_uint8_bgr(img_rgb_float)), conf=0.001, iou=0.7, max_det=300,
                         verbose=False, device=device)[0]
    xyxy = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
    return [{"image_id": int(image_id), "category_id": int(cat_ids[c]),
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls)]


def coco_eval(coco_gt, results, ids):
    """COCOeval stats on exactly these image ids (12 numbers; [0] = mAP@[.5:.95], [1] = mAP@.5)."""
    from pycocotools.cocoeval import COCOeval
    if not results:
        return np.zeros(12)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco_gt.loadRes(results)
        E = COCOeval(coco_gt, dt, "bbox"); E.params.imgIds = list(ids)
        E.evaluate(); E.accumulate(); E.summarize()
    return np.array(E.stats, dtype=float)


def per_image_ap(coco_gt, results, ids):
    """AP@[.5:.95] of each image on its own (notebooks 05 / 05b). An image with no detections gets 0."""
    by_img = {}
    for r in results:
        by_img.setdefault(r["image_id"], []).append(r)
    return {iid: float(coco_eval(coco_gt, by_img[iid], [iid])[0]) if by_img.get(iid) else 0.0 for iid in ids}
