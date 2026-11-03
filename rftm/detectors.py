"""Detectors for notebook 08's tests: YOLOv8n / YOLOv8s (ultralytics) and Faster R-CNN R50-FPN v2 (torchvision).
Each turns one float RGB image in [0, 1] into COCO result dicts, scored with the same pycocotools code as before.
Every image is quantised to uint8 first, exactly as notebooks 05/07 feed YOLO."""
import numpy as np
import torch

from . import coco as C


class UltralyticsDetector:
    def __init__(self, weights, coco_gt, device):
        from ultralytics import YOLO
        self.name = weights.replace(".pt", "")
        self.model = YOLO(weights)
        self.cat_ids = C.yolo_to_coco_ids(self.model, coco_gt)
        self.device = device

    def __call__(self, img, image_id):
        return C.detect(self.model, self.cat_ids, img, image_id, self.device)


class FasterRCNN:
    """torchvision's Faster R-CNN ResNet-50-FPN v2 (COCO, box mAP 46.7 on val2017 per torchvision).
    Its labels are COCO category ids directly (91 slots with 'N/A' gaps), checked against the annotations.
    Score threshold lowered to 0.001 and up to 300 boxes, like YOLO's mAP settings."""

    def __init__(self, coco_gt, device):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.name = "frcnn_r50_v2"
        self.model = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.001, box_detections_per_img=300).eval().to(device)
        cats = w.meta["categories"]
        bad = [(c, cats[c], coco_gt.loadCats(c)[0]["name"]) for c in coco_gt.getCatIds() if cats[c] != coco_gt.loadCats(c)[0]["name"]]
        assert not bad, f"torchvision / COCO category names disagree: {bad[:5]}"
        self.device = device

    @torch.no_grad()
    def __call__(self, img, image_id):
        u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1))).to(self.device).float() / 255.0
        out = self.model([t])[0]
        b = out["boxes"].cpu().numpy(); s = out["scores"].cpu().numpy(); l = out["labels"].cpu().numpy().astype(int)
        return [{"image_id": int(image_id), "category_id": int(c), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                 "score": float(sc)} for (x1, y1, x2, y2), sc, c in zip(b, s, l)]
