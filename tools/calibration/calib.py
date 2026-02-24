import pyk4a
from pyk4a import CalibrationType, PyK4A
import numpy as np
import time
from utils import decompose_homography, get_detections
import json
import cv2

cnt = pyk4a.connected_device_count()
print(f"Found {cnt} connected Azure Kinect devices.")

cfg = pyk4a.Config(
    color_resolution=pyk4a.ColorResolution.RES_1080P,
    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
    camera_fps=pyk4a.FPS.FPS_15,
    synchronized_images_only=False
)

camera_intrisics = {}
tag_to_global = None
camera_to_camera_transforms = None  # T_A_C = T_A_B @ inv(T_C_B)

for i in range(cnt):
    k4a = PyK4A(config=cfg, device_id=i)
    k4a.start()
    cap = None
    for attempt in range(10):  # Try up to 10 times
        cap = k4a.get_capture()
        if cap.color is not None:
            break
        print(f"  Waiting for color data... (attempt {attempt + 1})")
        time.sleep(0.5)
    cv2.imwrite(f"camera_{i}_capture.png", cap.color)  # Save the captured image for debugging
    calib = k4a.calibration
    K_color = calib.get_camera_matrix(CalibrationType.COLOR)
    camera_intrisics[i] = K_color

    color_image = cap.color
    detections = get_detections(color_image)
    if detections is None:
        print(f"  No detections found for camera {i}. Skipping...")
        continue
    det = detections[0] # Currently assuming we only have one tag in the frame
    H = det.homography.astype(np.float64)
    tag_to_cam = decompose_homography(H, K_color)
    if i == 0:
        tag_to_global = tag_to_cam
        camera_to_camera_transforms = {}
        camera_to_camera_transforms[i] = np.eye(4)  # Identity for the first camera
    else:
        cam_to_tag = np.linalg.inv(tag_to_cam)
        cam_to_world = tag_to_global @ cam_to_tag
        camera_to_camera_transforms[i] = cam_to_world

    k4a.stop()


with open("transformation_map.json", "w") as f:
    json.dump({str(k): v.tolist() for k, v in camera_to_camera_transforms.items()}, f)