import pyk4a
from pyk4a import CalibrationType, PyK4A
import numpy as np
import time
from utils import decompose_homography, get_detections
import cv2
import os
import pickle
from datetime import datetime

HALF_SIDE_M = 0.1  # tag half length in meters

print("Starting in 5 seconds...")
time.sleep(5)

cnt = pyk4a.connected_device_count()
print(f"Found {cnt} connected Azure Kinect devices.")

cfg = pyk4a.Config(
    # https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_gabd9688eb20d5cb878fd22d36de882ddb.html
    color_format=pyk4a.ImageFormat.COLOR_BGRA32,

    # https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_gabc7cab5e5396130f97b8ab392443c7b8.html
    color_resolution=pyk4a.ColorResolution.RES_1080P,

    # https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_ga3507ee60c1ffe1909096e2080dd2a05d.html
    depth_mode=pyk4a.DepthMode.OFF,

    camera_fps=pyk4a.FPS.FPS_15,
    synchronized_images_only=False
)

camera_intrisics = {}
tag_to_global = None
camera_to_camera_transforms = None  # T_A_C = T_A_B @ inv(T_C_B)

# output folders: calib_(DATE)/{cam_intr,cam_extr}/camera_<id>.pkl
date_str = datetime.now().strftime("%Y-%m-%d")
base_dir = f"calib_{date_str}"
intr_dir = os.path.join(base_dir, "cam_intr")
extr_dir = os.path.join(base_dir, "cam_extr")
os.makedirs(intr_dir, exist_ok=True)
os.makedirs(extr_dir, exist_ok=True)

for i in range(cnt):
    k4a = PyK4A(config=cfg, device_id=i)
    k4a.start()

    k4a._set_color_control(
        cmd=pyk4a.ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
        mode=pyk4a.ColorControlMode.MANUAL,
        value=8330
    )

    k4a._set_color_control(
        cmd=pyk4a.ColorControlCommand.WHITEBALANCE,
        mode=pyk4a.ColorControlMode.AUTO,
        value=0 # still required even though its auto
    )

    k4a._set_color_control(cmd=pyk4a.ColorControlCommand.BRIGHTNESS, mode=pyk4a.ColorControlMode.MANUAL, value=128)
    k4a._set_color_control(cmd=pyk4a.ColorControlCommand.CONTRAST, mode=pyk4a.ColorControlMode.MANUAL, value=5)
    k4a._set_color_control(cmd=pyk4a.ColorControlCommand.SATURATION, mode=pyk4a.ColorControlMode.MANUAL, value=32)
    k4a._set_color_control(cmd=pyk4a.ColorControlCommand.SHARPNESS, mode=pyk4a.ColorControlMode.MANUAL, value=2)
    k4a._set_color_control(cmd=pyk4a.ColorControlCommand.GAIN, mode=pyk4a.ColorControlMode.MANUAL, value=60)

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
    with open(os.path.join(intr_dir, f"camera_{i}.pkl"), "wb") as f:
        pickle.dump(K_color, f, protocol=pickle.HIGHEST_PROTOCOL)

    color_image = cap.color
    detections = get_detections(color_image)
    if detections is None:
        print(f"  No detections found for camera {i}. Skipping...")
        k4a.stop()
        continue

    det = detections[0]  # Currently assuming we only have one tag in the frame
    H = det.homography.astype(np.float64)
    tag_to_cam = decompose_homography(H, K_color)

    if i == 0:
        tag_to_global = tag_to_cam
        camera_to_camera_transforms = {}
        camera_to_camera_transforms[i] = np.eye(4)  # Identity for the first camera
        with open(os.path.join(extr_dir, f"camera_{i}.pkl"), "wb") as f:
            pickle.dump(camera_to_camera_transforms[i], f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        cam_to_tag = np.linalg.inv(tag_to_cam)
        cam_to_world = tag_to_global @ cam_to_tag
        camera_to_camera_transforms[i] = cam_to_world
        camera_to_camera_transforms[i][:3, 3] *= HALF_SIDE_M  # scale translation into meters;
        with open(os.path.join(extr_dir, f"camera_{i}.pkl"), "wb") as f:
            pickle.dump(camera_to_camera_transforms[i], f, protocol=pickle.HIGHEST_PROTOCOL)

    k4a.stop()