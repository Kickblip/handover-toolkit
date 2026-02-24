import numpy as np
import cv2
from pupil_apriltags import Detector

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Returns a list of april tag detections in a given image frame. 
def get_detections(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = at_detector.detect(gray)
    if len(detections) == 0:
        return None
    return detections

def decompose_homography(H, K):
    """
    Returns R (3x3) and t (3,) in tag units as a homogeneous transformation matrix.
    Assumes the planar model used to generate H has Z=0 and tag corners (±1, ±1).
    Takes in the homography matrix H (3x3) and the camera intrinsic matrix K (3x3).
    """
    Kinv = np.linalg.inv(K)
    A = Kinv @ H  # 3x3

    a1 = A[:, 0]
    a2 = A[:, 1]
    a3 = A[:, 2]

    # common scale (use average of norms; robust under noise)
    lam1 = 1.0 / np.linalg.norm(a1)
    lam2 = 1.0 / np.linalg.norm(a2)
    lam = 0.5 * (lam1 + lam2)

    r1 = lam * a1
    r2 = lam * a2
    r3 = np.cross(r1, r2)

    R = np.column_stack((r1, r2, r3))

    # Orthonormalize via SVD and enforce det=+1
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    t = lam * a3  # translation in "tag units"

    homogenous = np.eye(4)
    homogenous[:3,:3] = R
    homogenous[:3, 3] = t

    return homogenous.astype(np.float64)
