import os
from pathlib import Path
import argparse
import cv2
import numpy as np


def expand_and_clip_bbox(bbox, img_w, img_h, rescale_factor=2.0):
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1), (y2 - y1)
    s = max(w, h) * rescale_factor

    nx1, ny1 = cx - s / 2.0, cy - s / 2.0
    nx2, ny2 = cx + s / 2.0, cy + s / 2.0

    nx1 = int(max(0, min(img_w - 1, nx1)))
    ny1 = int(max(0, min(img_h - 1, ny1)))
    nx2 = int(max(0, min(img_w - 1, nx2)))
    ny2 = int(max(0, min(img_h - 1, ny2)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def detect_hand_boxes_from_vitpose(
    img_bgr, detector, vitpose, person_score_thr=0.5, kp_thr=0.5, rescale_factor=2.0
):
    H, W = img_bgr.shape[:2]

    # detect people (COCO class 0)
    det_out = detector(img_bgr)
    inst = det_out["instances"]
    valid = (inst.pred_classes == 0) & (inst.scores > person_score_thr)
    if valid.sum().item() == 0:
        return []

    person_boxes = inst.pred_boxes.tensor[valid].cpu().numpy()
    person_scores = inst.scores[valid].cpu().numpy()

    img_rgb = img_bgr[:, :, ::-1].copy()
    vitposes_out = vitpose.predict_pose(
        img_rgb,
        [np.concatenate([person_boxes, person_scores[:, None]], axis=1)],
    )

    results = []
    for person_id, vitposes in enumerate(vitposes_out):
        kpts = vitposes["keypoints"]  # (133,3) wholebody
        left = kpts[-42:-21]  # left hand 21 kpts
        right = kpts[-21:]  # right hand 21 kpts

        for is_right, hand_kp in [(0, left), (1, right)]:
            good = hand_kp[:, 2] > kp_thr
            if int(good.sum()) <= 3:
                continue

            xs = hand_kp[good, 0]
            ys = hand_kp[good, 1]
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            bbox_xyxy = expand_and_clip_bbox(bbox, W, H, rescale_factor=rescale_factor)
            if bbox_xyxy is None:
                continue

            results.append(
                {
                    "person_id": person_id,
                    "is_right": int(is_right),
                    "bbox_xyxy": bbox_xyxy.tolist(),
                }
            )

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="Directory containing input .jpg images")
    ap.add_argument("--out_dir", default="hand_crops_out", help="Where to write crops")
    ap.add_argument("--rescale_factor", type=float, default=2.0, help="Pad/expand bbox")
    ap.add_argument("--person_score", type=float, default=0.5, help="Person det score thresh")
    ap.add_argument("--kp_score", type=float, default=0.5, help="Hand kp score thresh")
    ap.add_argument("--detector", choices=["vitdet", "regnety"], default="vitdet")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    from htk.utils.utils_detectron2 import DefaultPredictor_Lazy

    if args.detector == "vitdet":
        from detectron2.config import LazyConfig
        import htk

        cfg_path = Path(htk.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg, device=args.device)
    else:
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg, device=args.device)

    from vitpose_model import ViTPoseModel

    vitpose = ViTPoseModel(args.device)

    img_dir = Path(args.image_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(str(img_dir))

    # Only .jpg images (case-insensitive)
    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg"]]
    )
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No .jpg/.jpeg images found in {img_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(str(img_path))

        boxes = detect_hand_boxes_from_vitpose(
            img_bgr,
            detector=detector,
            vitpose=vitpose,
            person_score_thr=args.person_score,
            kp_thr=args.kp_score,
            rescale_factor=args.rescale_factor,
        )

        # save crops + also dump bbox .npy files
        stem = img_path.stem
        for i, r in enumerate(boxes):
            x1, y1, x2, y2 = r["bbox_xyxy"]
            crop = img_bgr[y1:y2, x1:x2].copy()
            crop_name = f"{stem}_hand_{i:03d}_p{r['person_id']}_r{r['is_right']}.png"
            cv2.imwrite(os.path.join(args.out_dir, crop_name), crop)

            # bbox saved as [x1,y1,x2,y2]
            np.save(
                os.path.join(args.out_dir, f"{stem}_hand_{i:03d}_bbox.npy"),
                np.array([x1, y1, x2, y2], dtype=np.float32),
            )

        print(f"{img_path.name}: Found {len(boxes)} hands")
        for r in boxes:
            print(r)


if __name__ == "__main__":
    main()