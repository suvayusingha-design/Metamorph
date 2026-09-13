import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import os
import time
import json
import socket


# ============================================================
# DRONE AI SURVEILLANCE
# VIDEO PROTOTYPE
# ============================================================

print("=" * 60)
print("DRONE AI SURVEILLANCE")
print("VIDEO TEST MODE")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DROIDCAM LIVE CAMERA
# ============================================================
# DroidCam must be running on the phone and connected to the
# DroidCam Windows client. OpenCV sees DroidCam as a webcam.
#
# Previous setup used camera index 1, so keep that here.
# Camera index. Set to None to automatically find the first working webcam.
# If you know the correct index, use 0, 1, 2, etc.
DROIDCAM_INDEX = 1

# Windows camera backends to try. Some webcam drivers fail with DSHOW
# while working correctly with MSMF or the default OpenCV backend.
CAMERA_BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "visdrone_subset_5ep",
    "weights",
    "best.pt"
)

SAVE_OUTPUT = False
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "drone_surveillance_output.mp4"
)


# ============================================================
# YOLO SETTINGS
# ============================================================

# Lower confidence because aerial people are tiny.
CONFIDENCE = 0.15

IOU = 0.50

# Higher resolution helps small people.
INFERENCE_IMGSZ = 960

MAX_DET = 100

# Print detector diagnostics so inference errors are never silently hidden.
DEBUG_DETECTION = True


# ============================================================
# TILED DETECTION
# ============================================================

# Important for drone footage.
#
# Instead of:
#
#       FULL VIDEO FRAME
#              ↓
#            YOLO
#
# we use:
#
#       ┌───────┬───────┐
#       │ TILE  │ TILE  │
#       ├───────┼───────┤
#       │ TILE  │ TILE  │
#       └───────┴───────┘
#
# This makes small people much larger from YOLO's perspective.

USE_TILES = True

TILE_GRID = 2

# Small overlap prevents people sitting exactly on tile borders
# from being missed.
TILE_OVERLAP = 0.18


# ============================================================
# VIDEO PROCESSING
# ============================================================

# Process every frame.
#
# If your laptop becomes too slow, change this to 2.
PROCESS_EVERY_N_FRAMES = 1

# Loop the video instead of closing at the end.
LOOP_VIDEO = False

# Pause at the end if looping is disabled.
PAUSE_AT_END = True


# ============================================================
# GPS / ALTITUDE
# ============================================================

TEST_MODE = True

TEST_DRONE_LAT = 22.572600
TEST_DRONE_LON = 88.363900

# Prototype altitude.
#
# IMPORTANT:
# This is NOT calculated from the video.
# Later this should come from drone telemetry.
DRONE_ALTITUDE_M = 20.0


# ============================================================
# TARGET MODEL
# ============================================================

OVERHEAD_MODE = True


# ============================================================
# PERSISTENT TARGET
# ============================================================

# How long a target remains shown after disappearing.
OCCLUSION_TIMEOUT_SEC = 2.5

KEEP_TARGET_AFTER_TIMEOUT = False

# Size of occluded marker relative to previous target box.
OCCLUDED_BOX_SCALE = 0.60


# ============================================================
# UDP
# ============================================================

ENABLE_UDP = False

UDP_IP = "192.168.1.100"
UDP_PORT = 5005

udp_socket = None

if ENABLE_UDP:
    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )


# ============================================================
# COLORS
# ============================================================

GREEN = (60, 255, 60)
YELLOW = (0, 255, 255)
RED = (40, 40, 255)

WHITE = (240, 240, 240)
BLACK = (0, 0, 0)

DARK_GREEN = (15, 45, 15)

WINDOW_NAME = "DRONE AI SURVEILLANCE"


# ============================================================
# TRACKING DATA
# ============================================================

movement_history = defaultdict(list)
pose_history = defaultdict(list)

track_colors = {}

targets = {}


# ============================================================
# GPS
# ============================================================

def get_drone_gps():

    if TEST_MODE:

        return (
            TEST_DRONE_LAT,
            TEST_DRONE_LON,
            DRONE_ALTITUDE_M
        )

    raise RuntimeError(
        "Real GPS source has not been connected."
    )


# ============================================================
# TARGET GEOLOCATION
# ============================================================

def estimate_target_location(
    drone_lat,
    drone_lon,
    drone_alt
):

    # Prototype assumption:
    #
    # Drone is approximately directly above target.
    #
    # Therefore:
    #
    # Target GPS ≈ Drone GPS
    #
    # Target distance ≈ altitude

    return {

        "lat": float(drone_lat),

        "lon": float(drone_lon),

        "alt": 0.0,

        "distance": float(drone_alt)

    }


# ============================================================
# UDP
# ============================================================

def send_target(target):

    if udp_socket is None:
        return

    payload = {

        "id": target["id"],

        "latitude": target["lat"],

        "longitude": target["lon"],

        "altitude": target["alt"],

        "distance": target["distance"],

        "status": target["status"],

        "pose": target["pose"],

        "confidence": target["confidence"],

        "timestamp": time.time()

    }

    try:

        udp_socket.sendto(

            json.dumps(payload).encode(),

            (
                UDP_IP,
                UDP_PORT
            )

        )

    except OSError as e:

        print(
            "[UDP ERROR]",
            e
        )


# ============================================================
# TRACK COLORS
# ============================================================

def get_track_color(track_id):

    if track_id not in track_colors:

        rng = np.random.default_rng(
            int(track_id) * 777
        )

        track_colors[track_id] = tuple(

            int(
                rng.integers(
                    80,
                    255
                )
            )

            for _ in range(3)

        )

    return track_colors[track_id]


# ============================================================
# POSE HELPERS
# ============================================================

def kp_valid(
    kp,
    index,
    threshold=0.20
):

    return (

        kp is not None

        and index < len(kp)

        and float(
            kp[index][2]
        ) >= threshold

    )


def point(
    kp,
    index
):

    return np.array(

        [
            float(kp[index][0]),
            float(kp[index][1])
        ],

        dtype=np.float32

    )


def joint_angle(
    a,
    b,
    c
):

    ba = a - b
    bc = c - b

    denom = (

        np.linalg.norm(ba)

        *

        np.linalg.norm(bc)

    )

    if denom < 1e-6:
        return None

    cosine = np.clip(

        np.dot(ba, bc)

        /

        denom,

        -1.0,
        1.0

    )

    return float(

        np.degrees(

            np.arccos(cosine)

        )

    )


# ============================================================
# POSE CLASSIFICATION
# ============================================================

def classify_pose(
    kp,
    bbox
):

    if kp is None:
        return "UNKNOWN"

    x1, y1, x2, y2 = bbox

    width = max(
        1,
        x2 - x1
    )

    height = max(
        1,
        y2 - y1
    )

    aspect = width / height


    # --------------------------------------------------------
    # OBVIOUS LYING
    # --------------------------------------------------------

    if aspect > 1.45:
        return "LYING"


    # --------------------------------------------------------
    # KEYPOINTS
    # --------------------------------------------------------

    shoulders = [

        point(kp, i)

        for i in (5, 6)

        if kp_valid(kp, i)

    ]

    hips = [

        point(kp, i)

        for i in (11, 12)

        if kp_valid(kp, i)

    ]

    knees = [

        point(kp, i)

        for i in (13, 14)

        if kp_valid(kp, i)

    ]

    ankles = [

        point(kp, i)

        for i in (15, 16)

        if kp_valid(kp, i)

    ]


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not shoulders or not hips:

        if (
            aspect < 0.45
            and ankles
        ):
            return "STANDING"

        if aspect > 0.55:
            return "SITTING"

        return "UNKNOWN"


    # --------------------------------------------------------
    # TORSO
    # --------------------------------------------------------

    shoulder_center = np.mean(
        shoulders,
        axis=0
    )

    hip_center = np.mean(
        hips,
        axis=0
    )

    torso_dx = abs(

        float(
            hip_center[0]
            -
            shoulder_center[0]
        )

    )

    torso_dy = abs(

        float(
            hip_center[1]
            -
            shoulder_center[1]
        )

    )


    if torso_dx > torso_dy * 1.5:

        return "LYING"


    # --------------------------------------------------------
    # THIGH ORIENTATION
    # --------------------------------------------------------

    for hi, ki in (

        (11, 13),
        (12, 14)

    ):

        if (

            kp_valid(kp, hi)

            and

            kp_valid(kp, ki)

        ):

            hip = point(
                kp,
                hi
            )

            knee = point(
                kp,
                ki
            )

            dx = abs(
                float(
                    knee[0] -
                    hip[0]
                )
            )

            dy = abs(
                float(
                    knee[1] -
                    hip[1]
                )
            )

            if dx > dy * 0.70:

                return "SITTING"


    # --------------------------------------------------------
    # KNEE ANGLE
    # --------------------------------------------------------

    angles = []

    for hi, ki, ai in (

        (11, 13, 15),
        (12, 14, 16)

    ):

        if (

            kp_valid(kp, hi)

            and

            kp_valid(kp, ki)

            and

            kp_valid(kp, ai)

        ):

            angle = joint_angle(

                point(kp, hi),

                point(kp, ki),

                point(kp, ai)

            )

            if angle is not None:

                angles.append(angle)


    if angles:

        avg_angle = sum(
            angles
        ) / len(angles)

        if avg_angle < 125:
            return "SITTING"

        if avg_angle >= 145:
            return "STANDING"


    # --------------------------------------------------------
    # ANKLE POSITION
    # --------------------------------------------------------

    if ankles:

        ankle_y = float(

            np.mean(

                [
                    p[1]
                    for p in ankles
                ]

            )

        )

        ankle_drop = (

            ankle_y
            -
            hip_center[1]

        ) / height

        if ankle_drop > 0.45:

            return "STANDING"


    if aspect > 0.55:

        return "SITTING"


    return "UNKNOWN"


# ============================================================
# POSE SMOOTHING
# ============================================================

def smooth_pose(
    track_id,
    pose
):

    history = pose_history[
        track_id
    ]

    history.append(pose)

    if len(history) > 7:

        history.pop(0)

    useful = [

        p

        for p in history

        if p != "UNKNOWN"

    ]

    if not useful:
        return pose

    return Counter(
        useful
    ).most_common(1)[0][0]


# ============================================================
# DRAW CORNER BOX
# ============================================================

def draw_corner_box(
    frame,
    x1,
    y1,
    x2,
    y2,
    color,
    thickness=2
):

    width = max(
        10,
        x2 - x1
    )

    height = max(
        10,
        y2 - y1
    )

    corner = int(
        min(
            18,
            width * 0.25,
            height * 0.25
        )
    )

    # Top-left

    cv2.line(
        frame,
        (x1, y1),
        (x1 + corner, y1),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + corner),
        color,
        thickness
    )


    # Top-right

    cv2.line(
        frame,
        (x2 - corner, y1),
        (x2, y1),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + corner),
        color,
        thickness
    )


    # Bottom-left

    cv2.line(
        frame,
        (x1, y2 - corner),
        (x1, y2),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1 + corner, y2),
        color,
        thickness
    )


    # Bottom-right

    cv2.line(
        frame,
        (x2 - corner, y2),
        (x2, y2),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x2, y2 - corner),
        (x2, y2),
        color,
        thickness
    )


# ============================================================
# DASHED BOX
# ============================================================

def dashed_box(
    frame,
    x1,
    y1,
    x2,
    y2,
    color,
    thickness=2,
    dash=8
):

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        frame.shape[1] - 1,
        x2
    )

    y2 = min(
        frame.shape[0] - 1,
        y2
    )


    for x in range(
        x1,
        x2,
        dash * 2
    ):

        cv2.line(
            frame,
            (x, y1),
            (
                min(
                    x + dash,
                    x2
                ),
                y1
            ),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x, y2),
            (
                min(
                    x + dash,
                    x2
                ),
                y2
            ),
            color,
            thickness
        )


    for y in range(
        y1,
        y2,
        dash * 2
    ):

        cv2.line(
            frame,
            (x1, y),
            (
                x1,
                min(
                    y + dash,
                    y2
                )
            ),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x2, y),
            (
                x2,
                min(
                    y + dash,
                    y2
                )
            ),
            color,
            thickness
        )


# ============================================================
# OCCLUDED MARKER
# ============================================================

def draw_occluded_marker(
    frame,
    target
):

    x1 = target["last_bbox"][0]
    y1 = target["last_bbox"][1]
    x2 = target["last_bbox"][2]
    y2 = target["last_bbox"][3]

    cx = (
        x1 + x2
    ) // 2

    cy = (
        y1 + y2
    ) // 2


    width = max(
        20,
        int(
            (x2 - x1)
            *
            OCCLUDED_BOX_SCALE
        )
    )

    height = max(
        20,
        int(
            (y2 - y1)
            *
            OCCLUDED_BOX_SCALE
        )
    )


    nx1 = max(
        0,
        cx - width // 2
    )

    ny1 = max(
        0,
        cy - height // 2
    )

    nx2 = min(
        frame.shape[1] - 1,
        cx + width // 2
    )

    ny2 = min(
        frame.shape[0] - 1,
        cy + height // 2
    )


    dashed_box(
        frame,
        nx1,
        ny1,
        nx2,
        ny2,
        YELLOW,
        2,
        8
    )


    label = (
        f"UNKNOWN TARGET "
        f"{target['id']:02d}"
    )


    # Keep label inside frame.

    label_y = max(
        22,
        ny1 - 8
    )


    cv2.putText(
        frame,
        label,
        (
            nx1,
            label_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        YELLOW,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "OCCLUDED",
        (
            nx1,
            min(
                frame.shape[0] - 10,
                ny2 + 18
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        YELLOW,
        1,
        cv2.LINE_AA
    )


# ============================================================
# PANEL
# ============================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    title,
    lines,
    border=GREEN
):

    # Make sure panel stays inside frame.

    H, W = frame.shape[:2]

    x1 = max(
        5,
        min(x1, W - 10)
    )

    x2 = max(
        x1 + 10,
        min(x2, W - 5)
    )

    y1 = max(
        5,
        min(y1, H - 10)
    )

    y2 = max(
        y1 + 10,
        min(y2, H - 5)
    )


    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        DARK_GREEN,
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.58,
        frame,
        0.42,
        0
    )


    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        border,
        1
    )


    cv2.putText(
        frame,
        title,
        (
            x1 + 10,
            y1 + 22
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        border,
        1,
        cv2.LINE_AA
    )


    yy = y1 + 45

    for text in lines:

        cv2.putText(
            frame,
            text,
            (
                x1 + 10,
                yy
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            WHITE,
            1,
            cv2.LINE_AA
        )

        yy += 20


# ============================================================
# TILE GENERATOR
# ============================================================

def generate_tiles(
    frame
):

    H, W = frame.shape[:2]

    if not USE_TILES:

        return [

            (
                frame,
                0,
                0,
                W,
                H
            )

        ]


    tile_w = int(
        W / TILE_GRID
    )

    tile_h = int(
        H / TILE_GRID
    )

    overlap_x = int(
        tile_w *
        TILE_OVERLAP
    )

    overlap_y = int(
        tile_h *
        TILE_OVERLAP
    )


    tiles = []


    for row in range(
        TILE_GRID
    ):

        for col in range(
            TILE_GRID
        ):

            x1 = max(
                0,
                col * tile_w
                -
                overlap_x
            )

            y1 = max(
                0,
                row * tile_h
                -
                overlap_y
            )

            x2 = min(
                W,
                (col + 1) * tile_w
                +
                overlap_x
            )

            y2 = min(
                H,
                (row + 1) * tile_h
                +
                overlap_y
            )


            tile = frame[
                y1:y2,
                x1:x2
            ]


            tiles.append(

                (
                    tile,
                    x1,
                    y1,
                    x2,
                    y2
                )

            )


    return tiles


# ============================================================
# IOU
# ============================================================

def box_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )


    area_a = (
        max(
            0,
            ax2 - ax1
        )
        *
        max(
            0,
            ay2 - ay1
        )
    )


    area_b = (
        max(
            0,
            bx2 - bx1
        )
        *
        max(
            0,
            by2 - by1
        )
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:
        return 0.0


    return intersection / union


# ============================================================
# DETECTION
# ============================================================

def detect_people(
    model,
    frame
):
    """Detect people using YOLO11 Detect on overlapping tiles.

    IMPORTANT: this function intentionally does NOT use the pose model.
    Pose estimation is a separate second-stage operation and is disabled
    here until person detection is confirmed to work reliably.
    """

    all_detections = []
    tiles = generate_tiles(frame)
    tile_errors = 0
    raw_boxes = 0

    for tile_index, (tile, ox, oy, _, _) in enumerate(tiles):
        try:
            result = model.predict(
                tile,
                classes=[0],
                conf=CONFIDENCE,
                iou=IOU,
                imgsz=INFERENCE_IMGSZ,
                max_det=MAX_DET,
                verbose=False
            )[0]
        except Exception as e:
            tile_errors += 1
            print(f"[YOLO TILE ERROR {tile_index + 1}/{len(tiles)}] {type(e).__name__}: {e}")
            continue

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        raw_boxes += len(boxes)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])

            # Convert tile coordinates back to full-frame coordinates.
            x1 += ox
            x2 += ox
            y1 += oy
            y2 += oy

            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(x1 + 1, min(frame.shape[1] - 1, x2))
            y2 = max(y1 + 1, min(frame.shape[0] - 1, y2))

            # No keypoints at the detection stage.
            all_detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confs[i]),
                "keypoints": None
            })

    # --------------------------------------------------------
    # GLOBAL NMS AFTER TILE INFERENCE
    # --------------------------------------------------------
    # Overlapping tiles can produce the same person multiple
    # times.  Suppress those duplicates BEFORE tracking.
    all_detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    filtered = []
    NMS_IOU = 0.35

    for detection in all_detections:
        if any(
            box_iou(detection["bbox"], existing["bbox"]) >= NMS_IOU
            for existing in filtered
        ):
            continue
        filtered.append(detection)

    # Keep the detector from flooding the tracker with low-quality
    # boxes when a difficult aerial frame produces many candidates.
    filtered = filtered[:40]

    if DEBUG_DETECTION:
        print(
            f"[DETECT] tiles={len(tiles)} raw={raw_boxes} "
            f"final={len(filtered)} errors={tile_errors} "
            f"conf>={CONFIDENCE:.2f} imgsz={INFERENCE_IMGSZ}"
        )

    return filtered


# ============================================================
# STABLE TRACKER
# ============================================================

class SimpleTracker:
    """Small, conservative tracker for aerial person boxes.

    The previous tracker kept every historical track forever and used
    an 80-pixel minimum matching radius.  With overlapping drone tiles
    that caused new IDs to accumulate rapidly.
    """

    def __init__(self, max_missed=6):
        self.next_id = 1
        self.objects = {}
        self.max_missed = max_missed

    def update(self, detections):
        if not detections:
            for tid in list(self.objects):
                self.objects[tid]["missed"] += 1
                if self.objects[tid]["missed"] > self.max_missed:
                    del self.objects[tid]
            return []

        # Build candidate matches. Lower cost = better match.
        candidates = []
        for di, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dw = max(1.0, x2 - x1)
            dh = max(1.0, y2 - y1)

            for tid, obj in self.objects.items():
                if tid in {c[2] for c in candidates if c[0] == di}:
                    continue

                old_cx, old_cy = obj["center"]
                old_box = obj["bbox"]
                old_w = max(1.0, old_box[2] - old_box[0])
                old_h = max(1.0, old_box[3] - old_box[1])
                dist = float(np.hypot(cx - old_cx, cy - old_cy))
                iou = box_iou(detection["bbox"], old_box)

                # Small-object motion gate.  Crucially, this is not a
                # fixed 80-pixel radius.
                motion_gate = min(60.0, max(24.0, 0.90 * max(old_w, old_h, dw, dh)))

                if iou >= 0.10:
                    cost = (1.0 - iou) + 0.25 * min(dist / motion_gate, 2.0)
                    candidates.append((cost, di, tid))
                elif dist <= motion_gate:
                    size_ratio = max(dw / old_w, old_w / dw, dh / old_h, old_h / dh)
                    if size_ratio <= 2.2:
                        cost = 1.0 + dist / motion_gate
                        candidates.append((cost, di, tid))

        # Greedy one-to-one assignment.
        candidates.sort(key=lambda x: x[0])
        assigned_dets = set()
        assigned_ids = set()
        matches = {}

        for cost, di, tid in candidates:
            if di in assigned_dets or tid in assigned_ids:
                continue
            if cost > 2.0:
                continue
            assigned_dets.add(di)
            assigned_ids.add(tid)
            matches[di] = tid

        results = []

        for di, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if di in matches:
                tid = matches[di]
            else:
                tid = self.next_id
                self.next_id += 1

            self.objects[tid] = {
                "center": (cx, cy),
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "keypoints": detection.get("keypoints"),
                "missed": 0,
            }

            results.append({
                "id": tid,
                **detection
            })

        # Age tracks that were not matched this frame.
        for tid in list(self.objects):
            if tid not in assigned_ids and tid not in matches.values():
                self.objects[tid]["missed"] += 1
                if self.objects[tid]["missed"] > self.max_missed:
                    del self.objects[tid]

        return results


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            "[ERROR] MODEL NOT FOUND"
        )
        print()
        print(
            "Expected:"
        )
        print(
            MODEL_PATH
        )
        print()

        input(
            "Press ENTER to exit..."
        )

        return


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "Loading YOLO11 Detect (person detector)..."
    )

    try:

        model = YOLO(
            MODEL_PATH
        )

    except Exception as e:

        print()
        print(
            "[ERROR] MODEL LOAD FAILED"
        )
        print(
            e
        )

        input(
            "Press ENTER to exit..."
        )

        return


    print(
        "Model loaded."
    )

    print(
        "Detector task:",
        getattr(model, "task", "unknown")
    )


    # ========================================================
    # OPEN DROIDCAM
    # ========================================================

    print()
    print("Opening DroidCam...")
    print("Searching for a working Windows camera...")
    print()

    # Do not assume DroidCam is camera index 1. Windows can change
    # camera enumeration when devices are connected/disconnected.
    if DROIDCAM_INDEX is None:
        camera_indices = list(range(6))
    else:
        camera_indices = [DROIDCAM_INDEX]

    cap = None
    selected_index = None
    selected_backend_name = None

    for camera_index in camera_indices:
        for backend_name, backend in CAMERA_BACKENDS:
            print(
                f"Trying camera index {camera_index} "
                f"with {backend_name}..."
            )

            test_cap = cv2.VideoCapture(
                camera_index,
                backend
            )

            if not test_cap.isOpened():
                test_cap.release()
                continue

            # Test an actual frame. A backend can report opened while
            # still being unable to capture frames.
            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            test_cap.set(cv2.CAP_PROP_FPS, 30)

            ret, test_frame = test_cap.read()

            if ret and test_frame is not None:
                cap = test_cap
                selected_index = camera_index
                selected_backend_name = backend_name
                print()
                print(">>> CAMERA FOUND <<<")
                print(f"Index: {selected_index}")
                print(f"Backend: {selected_backend_name}")
                print()
                break

            test_cap.release()

        if cap is not None:
            break

    if cap is None:
        print()
        print("[ERROR] COULD NOT OPEN ANY CAMERA")
        print()
        print("Check:")
        print("1. DroidCam Windows Client is running.")
        print("2. DroidCam is connected to the phone.")
        print("3. Windows can see DroidCam as a camera.")
        print("4. Another app is not already using the camera.")
        print()
        print("Open Windows Camera and verify that DroidCam gives")
        print("a live picture before running this program again.")
        print()
        input("Press ENTER to exit...")
        return

    # Ask the selected camera for a practical live-video format.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0

    total_frames = 0

    print("DroidCam connected successfully.")
    print(f"Camera index: {selected_index}")
    print(f"Backend: {selected_backend_name}")
    print(
        "Camera:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    print("Camera FPS:", video_fps)
    print("Mode: LIVE DROIDCAM")
    print()
    # ========================================================
    # WINDOW
    # ========================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )


    # Start with a reasonable window size.

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        720
    )


    # ========================================================
    # TRACKER
    # ========================================================

    tracker = SimpleTracker()


    # ========================================================
    # STATE
    # ========================================================

    frame_number = 0

    paused = False

    last_time = time.time()
    start_time = last_time

    fps_display = 0.0

    writer = None

    last_detections = []

    print()
    print("Live DroidCam detector ready.")
    print()



    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # PAUSED
        # ----------------------------------------------------

        if paused:

            key = (
                cv2.waitKey(30)
                &
                0xFF
            )

            if key == ord("q"):

                break

            if key == 32:

                paused = False

            if key == ord("r"):

                # Live camera: reset the AI state, not the camera frame.
                frame_number = 0
                targets.clear()
                movement_history.clear()
                pose_history.clear()
                tracker = SimpleTracker()

                paused = False

            continue


        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()


        # ----------------------------------------------------
        # CAMERA FRAME LOSS / RECONNECT
        # ----------------------------------------------------

        if not ret:

            print("[WARNING] DroidCam frame lost. Reconnecting...")

            cap.release()
            time.sleep(0.5)

            # Reconnect using the backend/index that successfully worked,
            # then fall back to the automatic camera search.
            reconnect_attempts = [
                (selected_index, dict(CAMERA_BACKENDS)[selected_backend_name])
            ]

            if DROIDCAM_INDEX is None:
                reconnect_attempts = [
                    (idx, backend)
                    for idx in range(6)
                    for _, backend in CAMERA_BACKENDS
                ]

            new_cap = None

            for camera_index, backend in reconnect_attempts:
                test_cap = cv2.VideoCapture(camera_index, backend)

                if not test_cap.isOpened():
                    test_cap.release()
                    continue

                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                test_cap.set(cv2.CAP_PROP_FPS, 30)

                test_ret, test_frame = test_cap.read()

                if test_ret and test_frame is not None:
                    new_cap = test_cap
                    selected_index = camera_index
                    break

                test_cap.release()

            if new_cap is None:
                print("[WARNING] DroidCam reconnect failed. Retrying...")
                time.sleep(1.0)
                continue

            cap = new_cap
            print(
                f"[INFO] DroidCam reconnected "
                f"(index {selected_index})."
            )

            continue


        frame_number += 1


        H, W = frame.shape[:2]


        # ----------------------------------------------------
        # VIDEO TIME
        # ----------------------------------------------------

        video_time = time.time() - start_time


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        now = time.time()

        dt = now - last_time

        last_time = now

        if dt > 0:

            instant_fps = (
                1.0 / dt
            )

            if fps_display == 0:

                fps_display = (
                    instant_fps
                )

            else:

                fps_display = (

                    fps_display * 0.90

                    +

                    instant_fps * 0.10

                )


        # ----------------------------------------------------
        # GPS
        # ----------------------------------------------------

        drone_lat, drone_lon, drone_alt = (
            get_drone_gps()
        )


        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        if (

            frame_number %
            PROCESS_EVERY_N_FRAMES
            == 0

        ):

            try:

                detections = detect_people(

                    model,
                    frame

                )

                tracked = tracker.update(
                    detections
                )

                last_detections = tracked


            except Exception as e:

                print()
                print(
                    "[PROCESSING ERROR]"
                )
                print(
                    repr(e)
                )

                # Keep the current frame alive
                # rather than closing the window.

                tracked = []

        else:

            tracked = last_detections


        visible_ids = set()


        # ====================================================
        # VISIBLE TARGETS
        # ====================================================

        for detection in tracked:

            tid = detection["id"]

            x1, y1, x2, y2 = (
                detection["bbox"]
            )


            x1 = max(
                0,
                min(
                    W - 1,
                    x1
                )
            )

            y1 = max(
                0,
                min(
                    H - 1,
                    y1
                )
            )

            x2 = min(
                W - 1,
                max(
                    x1 + 1,
                    x2
                )
            )

            y2 = min(
                H - 1,
                max(
                    y1 + 1,
                    y2
                )
            )


            visible_ids.add(
                tid
            )


            center = (

                (x1 + x2) // 2,

                (y1 + y2) // 2

            )


            # ------------------------------------------------
            # POSE
            # ------------------------------------------------

            person_kp = (
                detection["keypoints"]
            )


            raw_pose = classify_pose(

                person_kp,

                (
                    x1,
                    y1,
                    x2,
                    y2
                )

            )


            pose = smooth_pose(
                tid,
                raw_pose
            )


            confidence = (

                detection[
                    "confidence"
                ]

                *

                100.0

            )


            # ------------------------------------------------
            # GEOLOCATION
            # ------------------------------------------------

            loc = estimate_target_location(

                drone_lat,
                drone_lon,
                drone_alt

            )


            # ------------------------------------------------
            # UPDATE TARGET
            # ------------------------------------------------

            targets[tid] = {

                "id":
                    tid,

                "lat":
                    loc["lat"],

                "lon":
                    loc["lon"],

                "alt":
                    loc["alt"],

                "distance":
                    loc["distance"],

                "status":
                    "VISIBLE",

                "pose":
                    pose,

                "confidence":
                    confidence,

                "last_seen_video_time":
                    video_time,

                "last_seen_frame":
                    frame_number,

                "last_pixel_center":
                    center,

                "last_bbox":
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )

            }


            send_target(
                targets[tid]
            )


            # =================================================
            # DRAW DETECTION
            # =================================================

            color = get_track_color(
                tid
            )


            draw_corner_box(

                frame,

                x1,
                y1,
                x2,
                y2,

                color,

                2

            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (

                f"UNKNOWN TARGET "
                f"{tid:02d}"

            )


            label_y = max(
                20,
                y1 - 8
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    label_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.43,

                color,

                1,

                cv2.LINE_AA

            )


            # ------------------------------------------------
            # INFO
            # ------------------------------------------------

            info = (

                f"{pose} | "
                f"{confidence:.0f}%"

            )


            info_y = min(

                H - 5,

                y2 + 16

            )


            cv2.putText(

                frame,

                info,

                (
                    x1,
                    info_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.37,

                color,

                1,

                cv2.LINE_AA

            )


            # ------------------------------------------------
            # CENTER DOT
            # ------------------------------------------------

            cv2.circle(

                frame,

                center,

                3,

                color,

                -1

            )


        # ====================================================
        # OCCLUDED TARGETS
        # ====================================================

        for tid in list(
            targets.keys()
        ):

            if tid in visible_ids:
                continue


            target = targets[
                tid
            ]


            elapsed = (

                video_time
                -
                target[
                    "last_seen_video_time"
                ]

            )


            if (

                elapsed
                <=
                OCCLUSION_TIMEOUT_SEC

            ):

                target[
                    "status"
                ] = "OCCLUDED"


                draw_occluded_marker(

                    frame,

                    target

                )


                send_target(
                    target
                )


            elif not KEEP_TARGET_AFTER_TIMEOUT:

                del targets[
                    tid
                ]


        # ====================================================
        # COUNTS
        # ====================================================

        visible_count = sum(

            1

            for target
            in targets.values()

            if target["status"]
            ==
            "VISIBLE"

        )


        occluded_count = sum(

            1

            for target
            in targets.values()

            if target["status"]
            ==
            "OCCLUDED"

        )


        # ====================================================
        # TOP BAR
        # ====================================================

        cv2.rectangle(

            frame,

            (
                0,
                0
            ),

            (
                W,
                45
            ),

            BLACK,

            -1

        )


        title = (
            "DRONE AI SURVEILLANCE"
        )


        cv2.putText(

            frame,

            title,

            (
                18,
                30
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            GREEN,

            2,

            cv2.LINE_AA

        )


        header = (

            f"DROIDCAM | YOLO11 DETECT | TILED | TRACKING | "
            f"{fps_display:.1f} FPS"

        )


        header_width = cv2.getTextSize(

            header,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            1

        )[0][0]


        cv2.putText(

            frame,

            header,

            (
                max(
                    450,
                    W -
                    header_width -
                    15
                ),

                29
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            GREEN,

            1,

            cv2.LINE_AA

        )


        # ====================================================
        # SYSTEM PANEL
        # ====================================================

        draw_panel(

            frame,

            15,
            60,
            300,
            250,

            "SYSTEM STATUS",

            [

                "VIDEO: LIVE DROIDCAM",

                "AI: YOLO11 DETECT",

                "TILED DETECTION: ON",

                "TRACKING: ACTIVE",

                "GEOLOCATION: ACTIVE",

                (
                    "GPS: TEST MODE"
                    if TEST_MODE
                    else
                    "GPS: LIVE"
                ),

                f"VISIBLE: {visible_count}",

                f"OCCLUDED: {occluded_count}",

            ]

        )


        # ====================================================
        # DRONE POSITION PANEL
        # ====================================================

        draw_panel(

            frame,

            15,
            265,
            300,
            425,

            "DRONE POSITION",

            [

                f"LAT: {drone_lat:.6f}",

                f"LON: {drone_lon:.6f}",

                f"ALT: {drone_alt:.1f} m",

                "MODE: OVERHEAD",

                f"DISTANCE: {drone_alt:.1f} m",

                "TARGET MODEL: GEO",

                "PERSISTENCE: ACTIVE",

            ]

        )


        # ====================================================
        # TARGET PANEL
        # ====================================================

        if targets:

            visible_targets = [

                t

                for t in targets.values()

                if t["status"]
                ==
                "VISIBLE"

            ]


            selected = (

                max(

                    visible_targets
                    or
                    list(
                        targets.values()
                    ),

                    key=lambda t:
                    t[
                        "last_seen_video_time"
                    ]

                )

            )


            border = (

                GREEN

                if selected[
                    "status"
                ]
                ==
                "VISIBLE"

                else

                YELLOW

            )


            panel_width = 340

            panel_height = 220


            panel_x2 = W - 15

            panel_x1 = max(

                330,

                panel_x2 -
                panel_width

            )


            panel_y2 = H - 20

            panel_y1 = max(

                455,

                panel_y2 -
                panel_height

            )


            draw_panel(

                frame,

                panel_x1,
                panel_y1,
                panel_x2,
                panel_y2,

                "PERSISTENT TARGET",

                [

                    (
                        f"ID: UNKNOWN TARGET "
                        f"{selected['id']:02d}"
                    ),

                    f"STATUS: {selected['status']}",

                    f"POSE: {selected['pose']}",

                    f"CONF: "
                    f"{selected['confidence']:.1f}%",

                    (
                        f"DISTANCE: "
                        f"{selected['distance']:.1f} m"
                    ),

                    (
                        f"LAT: "
                        f"{selected['lat']:.6f}"
                    ),

                    (
                        f"LON: "
                        f"{selected['lon']:.6f}"
                    ),

                    "AR MARKER: READY",

                ],

                border

            )


        # ====================================================
        # BOTTOM BAR
        # ====================================================

        cv2.putText(

            frame,

            f"TARGETS: {len(targets)}",

            (
                18,
                H - 18
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            GREEN,

            2,

            cv2.LINE_AA

        )


        controls = (
            "SPACE: PAUSE    "
            "R: RESET    "
            "Q: QUIT"
        )


        cv2.putText(

            frame,

            controls,

            (
                max(
                    400,
                    W - 300
                ),

                H - 18
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.35,

            WHITE,

            1,

            cv2.LINE_AA

        )


        # ====================================================
        # FRAME COUNTER
        # ====================================================

        frame_text = (

            f"FRAME: "
            f"{frame_number}"

        )


        cv2.putText(

            frame,

            frame_text,

            (
                W // 2 - 70,
                H - 18
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.35,

            WHITE,

            1,

            cv2.LINE_AA

        )


        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        if SAVE_OUTPUT:

            if writer is None:

                fourcc = (
                    cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )
                )

                writer = cv2.VideoWriter(

                    OUTPUT_PATH,

                    fourcc,

                    video_fps,

                    (
                        W,
                        H
                    )

                )


            writer.write(
                frame
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(

            WINDOW_NAME,

            frame

        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (

            cv2.waitKey(1)
            &
            0xFF

        )


        if key == ord("q"):

            print(
                "Q pressed. Exiting."
            )

            break


        elif key == 32:

            paused = not paused

            print(

                "PAUSED"
                if paused
                else
                "PLAYING"

            )


        elif key == ord("r"):

            print(
                "Resetting live camera tracking..."
            )

            frame_number = 0
            targets.clear()
            movement_history.clear()
            pose_history.clear()
            tracker = SimpleTracker()


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()


    if writer is not None:

        writer.release()


    if udp_socket is not None:

        udp_socket.close()


    cv2.destroyAllWindows()


    print()
    print("=" * 60)
    print(
        "DRONE AI SURVEILLANCE TERMINATED"
    )
    print("=" * 60)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Program interrupted."
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("FATAL ERROR")
        print("=" * 60)
        print(
            repr(e)
        )
        print()
        print(
            "The program did NOT silently close."
        )
        print(
            "Read the error above."
        )
        print("=" * 60)

        input(
            "Press ENTER to exit..."
        )