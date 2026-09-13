import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import threading
from gps_to_ar_hud_projection import (
    GPSPoint,
    project_target,
    draw_gps_target_box,
)

# ============================================================
# GOD'S EYE — GPS DATA BRIDGE
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
GPS_DATA_PATH = SCRIPT_DIR / "gps_data.json"
GPS_ALERT_PATH = SCRIPT_DIR / "gps_alert.json"

gps_lock = threading.Lock()


# ============================================================
# GPS DATA
# ============================================================

soldier_gps = {
    "latitude": 0.0,
    "longitude": 0.0,
    "accuracy": 0.0,
    "heading": 0.0,
    "received": False
}

person_gps = {
    "latitude": 0.0,
    "longitude": 0.0,
    "accuracy": 0.0,
    "received": False
}

_gps_missing_reported = False


# ============================================================
# SAFE FLOAT CONVERSION
# ============================================================

def safe_float(value, default=0.0):
    """
    Safely convert GPS values to float.

    Handles:
        None
        ""
        "--"
        "---"
        "N/A"
        "NA"
        "null"
        "None"
    """

    try:

        if value is None:
            return default

        if isinstance(value, str):

            value = value.strip()

            if value in (
                "",
                "--",
                "---",
                "N/A",
                "NA",
                "null",
                "None"
            ):
                return default

        return float(value)

    except (ValueError, TypeError):

        return default


# ============================================================
# READ GPS DATA
# ============================================================

def read_gps_data():
    """Read the latest phone data published by the GPS receiver."""

    global _gps_missing_reported

    if not GPS_DATA_PATH.exists():

        if not _gps_missing_reported:

            print()
            print(
                f"GPS file not found: {GPS_DATA_PATH}"
            )

            print(
                "Make sure gps_dual_receiver.py "
                "is running in the same folder."
            )

            _gps_missing_reported = True

        return

    _gps_missing_reported = False

    try:

        with GPS_DATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            return

        with gps_lock:

            # =================================================
            # SOLDIER
            # =================================================

            soldier = data.get(
                "soldier",
                {}
            )

            if isinstance(soldier, dict):

                soldier_gps["latitude"] = safe_float(
                    soldier.get("latitude"),
                    0.0
                )

                soldier_gps["longitude"] = safe_float(
                    soldier.get("longitude"),
                    0.0
                )

                soldier_gps["accuracy"] = safe_float(
                    soldier.get("accuracy"),
                    0.0
                )

                soldier_gps["heading"] = safe_float(
                    soldier.get("heading"),
                    0.0
                )

                soldier_gps["received"] = bool(
                    soldier.get(
                        "received",
                        False
                    )
                )

            # =================================================
            # TEST PERSON
            # =================================================

            person = data.get(
                "person",
                {}
            )

            if isinstance(person, dict):

                person_gps["latitude"] = safe_float(
                    person.get("latitude"),
                    0.0
                )

                person_gps["longitude"] = safe_float(
                    person.get("longitude"),
                    0.0
                )

                person_gps["accuracy"] = safe_float(
                    person.get("accuracy"),
                    0.0
                )

                person_gps["received"] = bool(
                    person.get(
                        "received",
                        False
                    )
                )

    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError
    ):

        # The receiver may be writing gps_data.json
        # at exactly the same time.
        #
        # Ignore the incomplete read and try again.
        pass


def read_gps_alert():
    """Read the latest GPS alignment result from gps_alert.json."""

    if not GPS_ALERT_PATH.exists():
        return {
            "gps_valid": False,
            "aligned": False,
            "bearing": None,
            "difference": None,
            "distance": None
        }

    try:
        with GPS_ALERT_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            return {
                "gps_valid": False,
                "aligned": False,
                "bearing": None,
                "difference": None,
                "distance": None
            }

        return data

    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError
    ):

        return {
            "gps_valid": False,
            "aligned": False,
            "bearing": None,
            "difference": None,
            "distance": None
        }


# ============================================================
# GPS BEARING CALCULATION
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):
    """Return compass bearing from Soldier to Test Person."""

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lon_rad = math.radians(
        lon2 - lon1
    )

    y = (
        math.sin(delta_lon_rad)
        * math.cos(lat2_rad)
    )

    x = (
        math.cos(lat1_rad)
        * math.sin(lat2_rad)

        -

        math.sin(lat1_rad)
        * math.cos(lat2_rad)
        * math.cos(delta_lon_rad)
    )

    return (
        math.degrees(
            math.atan2(y, x)
        )
        + 360.0
    ) % 360.0


# ============================================================
# GPS VALIDATION
# ============================================================

def gps_values_are_valid(
    soldier,
    person
):
    """Return True when both phones have usable GPS coordinates."""

    try:

        return (

            soldier.get(
                "received",
                False
            )

            and

            person.get(
                "received",
                False
            )

            and

            abs(
                float(
                    soldier.get(
                        "latitude",
                        0.0
                    )
                )
            ) > 0.000001

            and

            abs(
                float(
                    soldier.get(
                        "longitude",
                        0.0
                    )
                )
            ) > 0.000001

            and

            abs(
                float(
                    person.get(
                        "latitude",
                        0.0
                    )
                )
            ) > 0.000001

            and

            abs(
                float(
                    person.get(
                        "longitude",
                        0.0
                    )
                )
            ) > 0.000001
        )

    except (
        ValueError,
        TypeError
    ):

        return False


# ============================================================
# MR DIRECTION ALIGNMENT
# ============================================================

ALIGNMENT_TOLERANCE_DEGREES = 60.0

# GPS target screen projection calibration
CAMERA_HORIZONTAL_FOV_DEGREES = 70.0
CAMERA_HEADING_OFFSET_DEGREES = 0.0
CAMERA_X_DIRECTION = 1.0
TARGET_ELEVATION_DEGREES = 0.0
GPS_TARGET_BOX_W = 150
GPS_TARGET_BOX_H = 105


def angular_difference_degrees(
    a,
    b
):
    """Smallest absolute difference between compass headings."""

    return abs(
        (
            a - b + 180.0
        ) % 360.0
        - 180.0
    )


def signed_angular_difference_degrees(target, heading):
    """Signed target angle relative to camera heading."""
    return (target - heading + 180.0) % 360.0 - 180.0


def project_gps_target_to_screen(frame, target_bearing, camera_heading):
    """Project GPS target bearing into the camera frame."""
    if target_bearing is None or camera_heading is None:
        return None

    h, w = frame.shape[:2]
    signed_error = signed_angular_difference_degrees(
        target_bearing, camera_heading
    )
    half_fov = CAMERA_HORIZONTAL_FOV_DEGREES / 2.0

    if abs(signed_error) > half_fov:
        return None

    half_width = w / 2.0
    fx = half_width / math.tan(math.radians(half_fov))
    target_x = (
        half_width
        + CAMERA_X_DIRECTION * fx
        * math.tan(math.radians(signed_error))
    )
    target_y = h / 2.0

    box_w = GPS_TARGET_BOX_W
    box_h = GPS_TARGET_BOX_H
    min_x = box_w // 2 + 8
    max_x = w - box_w // 2 - 8
    target_x = max(min_x, min(max_x, target_x))

    return {
        "x": int(target_x),
        "y": int(target_y),
        "signed_error": signed_error,
        "box_w": box_w,
        "box_h": box_h
    }


# ============================================================
# GPS DISTANCE
# ============================================================

def calculate_distance_meters(
    lat1,
    lon1,
    lat2,
    lon2
):
    """Approximate surface distance between two GPS coordinates."""

    earth_radius = 6371000.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dlat / 2.0) ** 2

        +

        math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2.0) ** 2
    )

    c = (
        2.0
        * math.atan2(
            math.sqrt(a),
            math.sqrt(
                max(
                    0.0,
                    1.0 - a
                )
            )
        )
    )

    return earth_radius * c


# ============================================================
# GPS READER THREAD
# ============================================================

def gps_reader_loop():

    while True:

        read_gps_data()

        time.sleep(0.1)


threading.Thread(
    target=gps_reader_loop,
    daemon=True
).start()


print(
    "GPS data bridge started."
)

print(
    f"GPS file: {GPS_DATA_PATH}"
)

print(
    "Waiting for Soldier + Test Person data..."
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

MODEL_PATH = (
    SCRIPT_DIR
    / "models"
    / "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=str(
        MODEL_PATH
    )
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.60,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.60
)

detector = (
    vision.HandLandmarker
    .create_from_options(
        options
    )
)


# ============================================================
# YOLOv8n PERSON DETECTION
# ============================================================

YOLO_MODEL_PATH = (
    SCRIPT_DIR
    / "yolov8n.pt"
)

YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640

YOLO_MIN_PERSON_HEIGHT = 90
YOLO_MIN_PERSON_AREA_RATIO = 0.0035

MIN_INTERACTIVE_HAND_WIDTH = 65
MIN_INTERACTIVE_HAND_HEIGHT = 70

HAND_MASK_EXPANSION = 1.35


print(
    "Loading YOLOv8n..."
)

yolo_model = YOLO(
    str(
        YOLO_MODEL_PATH
    )
)

print(
    "YOLOv8n loaded successfully!"
)


detected_person_count = 0
highest_person_confidence = 0.0


# ============================================================
# HAND BOUNDING BOX
# ============================================================

def get_expanded_hand_bbox(
    hand,
    width,
    height
):
    """Return expanded pixel bbox around MediaPipe hand."""

    xs = [
        int(
            landmark.x * width
        )
        for landmark in hand
    ]

    ys = [
        int(
            landmark.y * height
        )
        for landmark in hand
    ]

    x1 = max(
        0,
        min(xs)
    )

    y1 = max(
        0,
        min(ys)
    )

    x2 = min(
        width - 1,
        max(xs)
    )

    y2 = min(
        height - 1,
        max(ys)
    )

    hand_w = max(
        1,
        x2 - x1
    )

    hand_h = max(
        1,
        y2 - y1
    )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    expanded_w = (
        hand_w
        * HAND_MASK_EXPANSION
    )

    expanded_h = (
        hand_h
        * HAND_MASK_EXPANSION
    )

    ex1 = max(
        0,
        int(
            cx
            - expanded_w / 2
        )
    )

    ey1 = max(
        0,
        int(
            cy
            - expanded_h / 2
        )
    )

    ex2 = min(
        width - 1,
        int(
            cx
            + expanded_w / 2
        )
    )

    ey2 = min(
        height - 1,
        int(
            cy
            + expanded_h / 2
        )
    )

    return (
        ex1,
        ey1,
        ex2,
        ey2,
        hand_w,
        hand_h
    )


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def run_person_detection(
    frame
):
    """Return filtered YOLOv8 person detections."""

    detections = []

    results = yolo_model.predict(
        source=frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        classes=[0],
        verbose=False
    )

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    frame_h, frame_w = (
        frame.shape[:2]
    )

    frame_area = (
        frame_w * frame_h
    )

    for box in result.boxes:

        coords = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        confidence = float(
            box.conf[0]
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = (
            coords.tolist()
        )

        x1 = max(
            0,
            min(
                frame_w - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                frame_h - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                frame_w - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                frame_h - 1,
                y2
            )
        )

        box_w = max(
            0,
            x2 - x1
        )

        box_h = max(
            0,
            y2 - y1
        )

        box_area = (
            box_w * box_h
        )

        area_ratio = (
            box_area / frame_area
            if frame_area
            else 0
        )

        if box_h < YOLO_MIN_PERSON_HEIGHT:
            continue

        if area_ratio < YOLO_MIN_PERSON_AREA_RATIO:
            continue

        # --------------------------------------------------------
        # SIMPLE BLACK-CLOTHING HEURISTIC
        # Uses the central upper-body region only. This is NOT an
        # enemy classifier; non-black clothing remains UNKNOWN.
        # --------------------------------------------------------
        upper_y1 = y1 + int(box_h * 0.20)
        upper_y2 = y1 + int(box_h * 0.62)
        upper_x1 = x1 + int(box_w * 0.20)
        upper_x2 = x1 + int(box_w * 0.80)

        upper_y1 = max(0, min(frame_h - 1, upper_y1))
        upper_y2 = max(0, min(frame_h, upper_y2))
        upper_x1 = max(0, min(frame_w - 1, upper_x1))
        upper_x2 = max(0, min(frame_w, upper_x2))

        black_ratio = 0.0

        if upper_x2 > upper_x1 and upper_y2 > upper_y1:
            clothing_crop = frame[upper_y1:upper_y2, upper_x1:upper_x2]
            hsv_crop = cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2HSV)

            # Low-value pixels are treated as visually black.
            black_mask = cv2.inRange(
                hsv_crop,
                np.array([0, 0, 0], dtype=np.uint8),
                np.array([180, 255, 65], dtype=np.uint8)
            )

            black_ratio = float(np.count_nonzero(black_mask)) / float(black_mask.size)

        clothing_status = (
            "FRIENDLY - BLACK CLOTHING"
            if black_ratio >= 0.45
            else "UNKNOWN - OTHER CLOTHING"
        )

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "black_ratio": black_ratio,
                "clothing_status": clothing_status
            }
        )

    return detections


# ============================================================
# DRAW YOLO PERSON DETECTIONS
# ============================================================

def draw_person_detections(
    frame,
    detections
):
    """Draw green bounding boxes around detected people."""

    for detection in detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        clothing_status = detection.get(
            "clothing_status",
            "UNKNOWN - OTHER CLOTHING"
        )

        # Friendly (black clothing) gets a different box colour.
        # Other detections retain the existing green box.
        box_color = (
            BLUE
            if clothing_status == "FRIENDLY - BLACK CLOTHING"
            else GREEN
        )

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            box_color,
            2
        )

        # Keep the existing person box and add a compact clothing label.
        label_y = max(24, y1 - 8)

        cv2.putText(
            frame,
            clothing_status,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            box_color,
            2,
            cv2.LINE_AA
        )


# ============================================================
# DROIDCAM
# ============================================================

DROIDCAM_CAMERA_INDEX = 1


def open_droidcam_camera():
    """Open DroidCam using Windows-compatible backends."""

    attempts = [

        (
            "default",

            lambda:
            cv2.VideoCapture(
                DROIDCAM_CAMERA_INDEX
            )
        ),

        (
            "MSMF",

            lambda:
            cv2.VideoCapture(
                DROIDCAM_CAMERA_INDEX,
                cv2.CAP_MSMF
            )
        )
    ]

    for (
        name,
        opener
    ) in attempts:

        print(
            f"Trying DroidCam camera "
            f"{DROIDCAM_CAMERA_INDEX} "
            f"using {name}..."
        )

        cam = None

        try:

            cam = opener()

            if not cam.isOpened():

                if cam is not None:
                    cam.release()

                continue

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"MJPG"
                )
            )

            cam.set(
                cv2.CAP_PROP_FOURCC,
                fourcc
            )

            cam.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            cam.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

            cam.set(
                cv2.CAP_PROP_FPS,
                30
            )

            cam.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            valid_frame = False

            for _ in range(30):

                ok, test_frame = (
                    cam.read()
                )

                if (
                    ok
                    and test_frame is not None
                ):

                    valid_frame = True

                    break

            if valid_frame:

                actual_width = int(
                    cam.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                actual_height = int(
                    cam.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                print(
                    f"DroidCam connected: "
                    f"{actual_width}x"
                    f"{actual_height}"
                )

                return cam

            print(
                f"{name} opened the camera "
                f"but returned no frame."
            )

            cam.release()

        except Exception as exc:

            print(
                f"{name} failed:",
                exc
            )

            if cam is not None:
                cam.release()

    return None


# ============================================================
# START DROIDCAM
# ============================================================

print(
    "Starting DroidCam camera..."
)

camera = (
    open_droidcam_camera()
)

if camera is None:

    print(
        "\nERROR: DroidCam camera "
        "could not be opened."
    )

    print(
        "\nDo this in the DroidCam PC Client:"
    )

    print(
        "1. Open the DroidCam source Properties."
    )

    print(
        "2. Set Video Format to MJPEG."
    )

    print(
        "3. Set video size to 1280x720."
    )

    print(
        "4. Deactivate and Activate the source again."
    )

    print(
        "5. Then run this program again."
    )

    detector.close()

    raise SystemExit


print(
    "DroidCam camera connected successfully!"
)


# ============================================================
# GOD'S EYE SETTINGS
# ============================================================

PINCH_RATIO_THRESHOLD = 0.65
pinch_was_active = False


# ============================================================
# PANEL STATES
# ============================================================

panels = {

    "drone_feed": True,

    "detection": True,

    "map": True,

    "drone_status": True,

    "target_info": True,

    "system_status": True
}


# ============================================================
# COLORS — OpenCV BGR
# ============================================================

WHITE = (
    255,
    255,
    255
)

GREEN = (
    0,
    255,
    0
)

RED = (
    0,
    0,
    255
)

BLUE = (
    255,
    180,
    0
)


# ============================================================
# MR ALIGNMENT
# ============================================================

def draw_mr_alignment(
    frame,
    aligned,
    bearing,
    heading,
    error,
    distance,
    target_screen=None
):

    h, w = (
        frame.shape[:2]
    )

    cx = w // 2
    cy = h // 2


    # ========================================================
    # CENTER RETICLE
    # ========================================================

    cv2.circle(
        frame,
        (
            cx,
            cy
        ),
        7,
        WHITE,
        1
    )

    cv2.line(
        frame,
        (
            cx - 16,
            cy
        ),
        (
            cx - 5,
            cy
        ),
        WHITE,
        1
    )

    cv2.line(
        frame,
        (
            cx + 5,
            cy
        ),
        (
            cx + 16,
            cy
        ),
        WHITE,
        1
    )

    cv2.line(
        frame,
        (
            cx,
            cy - 16
        ),
        (
            cx,
            cy - 5
        ),
        WHITE,
        1
    )

    cv2.line(
        frame,
        (
            cx,
            cy + 5
        ),
        (
            cx,
            cy + 16
        ),
        WHITE,
        1
    )


    # ========================================================
    # GPS-PREDICTED TARGET — UNIDENTIFIED PERSON
    # ========================================================
    # Show the GPS target only when it is aligned and YOLO has not
    # already detected a person. It uses the same normal rectangular
    # detection-box appearance.
    if aligned and target_screen is not None:

        target_x = target_screen["x"]
        target_y = target_screen["y"]
        box_w = target_screen["box_w"]
        box_h = target_screen["box_h"]

        x1 = max(0, target_x - box_w // 2)
        y1 = max(0, target_y - box_h // 2)
        x2 = min(w - 1, target_x + box_w // 2)
        y2 = min(h - 1, target_y + box_h // 2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            GREEN,
            2
        )

        cv2.putText(
            frame,
            "UNIDENTIFIED PERSON SPOTTED",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            GREEN,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # GPS DIAGNOSTIC TEXT
    # ========================================================

    if bearing is not None:

        cv2.putText(
            frame,
            f"BEARING {bearing:.1f}°",
            (
                cx - 95,
                82
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1
        )

    if heading is not None:

        cv2.putText(
            frame,
            f"HEADING {heading:.1f}°",
            (
                cx - 95,
                102
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1
        )

    if error is not None:

        cv2.putText(
            frame,
            f"ERROR {error:.1f}°",
            (
                cx - 95,
                122
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            GREEN if aligned else WHITE,
            1
        )

    if distance is not None:

        distance_text = (

            f"DIST {distance:.1f} m"

            if distance < 1000

            else
            f"DIST {distance / 1000:.2f} km"
        )

        cv2.putText(
            frame,
            distance_text,
            (
                cx - 95,
                142
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1
        )


def draw_gps_alignment_alert(
    frame,
    gps_alert
):
    """Display GPS alignment alert on the HUD."""

    if not gps_alert.get("gps_valid", False):
        return

    if not gps_alert.get("aligned", False):
        return

    h, w = frame.shape[:2]

    alert_text = "UNIDENTIFIED PERSON SPOTTED"

    text_size = cv2.getTextSize(
        alert_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        2
    )[0]

    text_x = (
        w // 2
        - text_size[0] // 2
    )

    text_y = 185

    cv2.putText(
        frame,
        alert_text,
        (
            text_x,
            text_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        GREEN,
        2
    )

    distance = gps_alert.get("distance")

    if distance is not None:
        distance_text = (
            f"DIST  {distance:.1f} m"
            if distance < 1000
            else f"DIST  {distance / 1000:.2f} km"
        )

        cv2.putText(
            frame,
            distance_text,
            (
                text_x,
                text_y + 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1
        )


# ============================================================
# CORNER BOX
# ============================================================

def draw_corner_box(
    frame,
    x,
    y,
    w,
    h,
    color=WHITE,
    thickness=2
):

    corner = 18

    cv2.line(
        frame,
        (
            x,
            y
        ),
        (
            x + corner,
            y
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x,
            y
        ),
        (
            x,
            y + corner
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x + w - corner,
            y
        ),
        (
            x + w,
            y
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x + w,
            y
        ),
        (
            x + w,
            y + corner
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x,
            y + h - corner
        ),
        (
            x,
            y + h
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x,
            y + h
        ),
        (
            x + corner,
            y + h
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x + w - corner,
            y + h
        ),
        (
            x + w,
            y + h
        ),
        color,
        thickness
    )

    cv2.line(
        frame,
        (
            x + w,
            y + h - corner
        ),
        (
            x + w,
            y + h
        ),
        color,
        thickness
    )


# ============================================================
# CONTROL BUTTON
# ============================================================

def draw_control_button(
    frame,
    x,
    y,
    visible=True,
    hovered=False
):

    radius = 13

    color = (
        GREEN
        if hovered
        else WHITE
    )

    symbol = (
        "-"
        if visible
        else "+"
    )

    cv2.circle(
        frame,
        (
            x,
            y
        ),
        radius,
        color,
        2
    )

    cv2.putText(
        frame,
        symbol,
        (
            x - 7,
            y + 6
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


# ============================================================
# DRAW PANEL
# ============================================================

def draw_panel(
    frame,
    x,
    y,
    w,
    h,
    title,
    panel_key,
    side="left"
):

    visible = panels[
        panel_key
    ]

    button_x = (
        x + w
        if side == "left"
        else x
    )

    button_y = (
        y + 20
    )

    if not visible:

        draw_control_button(
            frame,
            button_x,
            button_y,
            visible=False
        )

        return

    draw_corner_box(
        frame,
        x,
        y,
        w,
        h,
        WHITE,
        2
    )

    draw_control_button(
        frame,
        button_x,
        button_y,
        visible=True
    )

    cv2.putText(
        frame,
        title,
        (
            x + 15,
            y + 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2
    )

    cv2.line(
        frame,
        (
            x + 12,
            y + 38
        ),
        (
            x + w - 12,
            y + 38
        ),
        WHITE,
        1
    )


# ============================================================
# PANEL CONTENT
# ============================================================

def draw_panel_content(
    frame,
    x,
    y,
    w,
    h,
    panel_key
):

    # ========================================================
    # MAP — PROMINENT LIVE GPS DISPLAY
    # ========================================================

    if panel_key == "map":

        person_received = (
            current_person.get(
                "received",
                False
            )
        )

        if person_received:

            person_lat = safe_float(
                current_person.get(
                    "latitude"
                ),
                0.0
            )

            person_lon = safe_float(
                current_person.get(
                    "longitude"
                ),
                0.0
            )

            lat_text = (
                f"{person_lat:.6f}"
            )

            lon_text = (
                f"{person_lon:.6f}"
            )

        else:

            lat_text = "--.------"
            lon_text = "--.------"


        # ====================================================
        # LIVE TARGET INDICATOR
        # ====================================================

        cv2.circle(
            frame,
            (
                x + 19,
                y + 66
            ),
            4,
            GREEN if person_received else WHITE,
            -1
        )

        cv2.putText(
            frame,
            "LIVE TARGET",
            (
                x + 30,
                y + 70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            GREEN if person_received else WHITE,
            1
        )


        # ====================================================
        # LATITUDE LABEL
        # ====================================================

        cv2.putText(
            frame,
            "LATITUDE",
            (
                x + 15,
                y + 91
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            WHITE,
            1
        )


        # ====================================================
        # LATITUDE VALUE
        # ====================================================

        cv2.putText(
            frame,
            lat_text,
            (
                x + 15,
                y + 111
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            GREEN if person_received else WHITE,
            2
        )


        # ====================================================
        # LONGITUDE LABEL
        # ====================================================

        cv2.putText(
            frame,
            "LONGITUDE",
            (
                x + 15,
                y + 130
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            WHITE,
            1
        )


        # ====================================================
        # LONGITUDE VALUE
        # ====================================================

        cv2.putText(
            frame,
            lon_text,
            (
                x + 15,
                y + 147
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            GREEN if person_received else WHITE,
            2
        )

        return


    # ========================================================
    # NORMAL PANEL CONTENT
    # ========================================================

    panel_content = {

        "drone_feed": (
            "LIVE FEED",
            "SIGNAL: 98%",
            "ALT: 120 m"
        ),

        "detection": (
            f"TARGETS: {detected_person_count}",
            "SCANNING...",
            "STATUS: CLEAR"
        ),

        "drone_status": (
            "ONLINE",
            "BATTERY: 87%",
            "LINK: STABLE"
        ),

        "target_info": (
            f"YOLO PERSONS: {detected_person_count}",
            (
                "GPS: TARGET IN FOV"
                if target_screen is not None
                else "GPS: OUTSIDE FOV"
            ),
            (
                f"BRG {bearing_text} | DELTA {signed_target_delta:+.1f}°"
                if signed_target_delta is not None
                else "BRG -- | DELTA --"
            )
        ),

        "system_status": (
            "SYSTEM: READY",
            "CAMERA: OK",
            "YOLO: ACTIVE"
        )

    }.get(
        panel_key,
        ()
    )


    # ========================================================
    # DRAW NORMAL CONTENT
    # ========================================================

    for index, text in enumerate(
        panel_content
    ):

        cv2.putText(
            frame,
            text,
            (
                x + 15,
                y + 68 + index * 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            GREEN if index == 0 else WHITE,
            1
        )


# ============================================================
# PANEL POSITIONS
# ============================================================

PANEL_W = 250
PANEL_H = 145

LEFT_X = 35

ROW1_Y = 120
ROW2_Y = 295
ROW3_Y = 470


# ============================================================
# GPS TERMINAL PRINT TIMER
# ============================================================

last_gps_terminal_print = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = (
        camera.read()
    )

    if (
        not success
        or frame is None
    ):

        print(
            "\nCould not read a frame "
            "from DroidCam."
        )

        break


    # ========================================================
    # LIVE GPS DATA
    # ========================================================

    with gps_lock:

        current_soldier = (
            soldier_gps.copy()
        )

        current_person = (
            person_gps.copy()
        )

    gps_alert = read_gps_alert()


    # ========================================================
    # DEFAULT GPS VALUES
    # ========================================================

    target_bearing = None
    target_distance = None
    heading_error = None
    signed_target_delta = None
    camera_heading = None
    target_screen = None
    direction_aligned = False


    # ========================================================
    # CALCULATE TARGET BEARING
    # ========================================================

    if gps_values_are_valid(
        current_soldier,
        current_person
    ):

        target_bearing = (
            calculate_bearing(
                current_soldier["latitude"],
                current_soldier["longitude"],
                current_person["latitude"],
                current_person["longitude"]
            )
        )

        target_distance = (
            calculate_distance_meters(
                current_soldier["latitude"],
                current_soldier["longitude"],
                current_person["latitude"],
                current_person["longitude"]
            )
        )

        soldier_heading = (
            float(
                current_soldier["heading"]
            )
            % 360.0
        )

        camera_heading = (
            soldier_heading
            + CAMERA_HEADING_OFFSET_DEGREES
        ) % 360.0

        heading_error = (
            angular_difference_degrees(
                camera_heading,
                target_bearing
            )
        )

        signed_target_delta = (
            signed_angular_difference_degrees(
                target_bearing,
                camera_heading
            )
        )

        direction_aligned = (
            heading_error
            <= ALIGNMENT_TOLERANCE_DEGREES
        )

        target_screen = project_gps_target_to_screen(
            frame,
            target_bearing,
            camera_heading
        )

    # ========================================================
    # BEARING TEXT
    # ========================================================

    bearing_text = (

        f"{target_bearing:.1f}°"

        if target_bearing is not None

        else "--"
    )


    # ========================================================
    # GPS TERMINAL STATUS
    # ========================================================
    #
    # Print once per second.
    # Prevents terminal flooding.
    # ========================================================

    current_time = time.time()

    if (
        current_time
        - last_gps_terminal_print
        >= 1.0
    ):

        last_gps_terminal_print = (
            current_time
        )

        if gps_values_are_valid(
            current_soldier,
            current_person
        ):

            error_text = (

                f"{heading_error:.1f}°"

                if heading_error is not None

                else "--"
            )

            print(

                "\nGPS OK | "

                f"Soldier: "
                f"{current_soldier['latitude']:.6f}, "
                f"{current_soldier['longitude']:.6f} | "

                f"Heading: "
                f"{current_soldier['heading']:.1f}° | "

                f"Target: "
                f"{current_person['latitude']:.6f}, "
                f"{current_person['longitude']:.6f} | "

                f"Bearing: "
                f"{bearing_text} | "

                f"Error: "
                f"{error_text}"
            )

        else:

            print(

                "\nGPS WAITING | "

                f"Soldier received: "
                f"{current_soldier.get('received', False)} | "

                f"Person received: "
                f"{current_person.get('received', False)}"
            )


    # ========================================================
    # FRAME SIZE
    # ========================================================

    height, width, _ = (
        frame.shape
    )

    right_x = (
        width
        - PANEL_W
        - 35
    )


    # ========================================================
    # PANEL DEFINITIONS
    # ========================================================

    panel_definitions = [

        (
            "drone_feed",
            LEFT_X,
            ROW1_Y,
            "left"
        ),

        (
            "detection",
            LEFT_X,
            ROW2_Y,
            "left"
        ),

        (
            "map",
            LEFT_X,
            ROW3_Y,
            "left"
        ),

        (
            "drone_status",
            right_x,
            ROW1_Y,
            "right"
        ),

        (
            "target_info",
            right_x,
            ROW2_Y,
            "right"
        ),

        (
            "system_status",
            right_x,
            ROW3_Y,
            "right"
        )
    ]


    # ========================================================
    # MEDIAPIPE HAND TRACKING
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(
        mp_image
    )

    pinching_hands = []

    hand_mask_boxes = []


    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            (
                mask_x1,
                mask_y1,
                mask_x2,
                mask_y2,
                detected_hand_width,
                detected_hand_height
            ) = get_expanded_hand_bbox(
                hand,
                width,
                height
            )

            hand_mask_boxes.append(
                (
                    mask_x1,
                    mask_y1,
                    mask_x2,
                    mask_y2
                )
            )


            # =================================================
            # IGNORE SMALL/FAR HANDS
            # =================================================

            if (
                detected_hand_width
                < MIN_INTERACTIVE_HAND_WIDTH

                or

                detected_hand_height
                < MIN_INTERACTIVE_HAND_HEIGHT
            ):

                continue


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for landmark in hand:

                px = int(
                    landmark.x
                    * width
                )

                py = int(
                    landmark.y
                    * height
                )

                cv2.circle(
                    frame,
                    (
                        px,
                        py
                    ),
                    2,
                    GREEN,
                    -1
                )


            # =================================================
            # INDEX + THUMB
            # =================================================

            index_tip = hand[8]
            thumb_tip = hand[4]

            index_x = int(
                index_tip.x
                * width
            )

            index_y = int(
                index_tip.y
                * height
            )

            thumb_x = int(
                thumb_tip.x
                * width
            )

            thumb_y = int(
                thumb_tip.y
                * height
            )

            cv2.circle(
                frame,
                (
                    index_x,
                    index_y
                ),
                8,
                RED,
                -1
            )

            cv2.circle(
                frame,
                (
                    thumb_x,
                    thumb_y
                ),
                7,
                BLUE,
                -1
            )


            # =================================================
            # PINCH CALCULATION
            # =================================================

            index_mcp = hand[5]
            pinky_mcp = hand[17]

            ix = (
                index_mcp.x
                * width
            )

            iy = (
                index_mcp.y
                * height
            )

            px = (
                pinky_mcp.x
                * width
            )

            py = (
                pinky_mcp.y
                * height
            )

            hand_width = math.hypot(
                ix - px,
                iy - py
            )

            pinch_distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            pinch_ratio = (
                pinch_distance
                / hand_width
                if hand_width > 1
                else 999
            )

            is_pinching = (
                pinch_ratio
                < PINCH_RATIO_THRESHOLD
            )

            pinching_hands.append(
                {
                    "index_x": index_x,
                    "index_y": index_y,
                    "pinching": is_pinching
                }
            )


    # ========================================================
    # YOLO INPUT
    # ========================================================

    yolo_input = frame.copy()

    for (
        mask_x1,
        mask_y1,
        mask_x2,
        mask_y2
    ) in hand_mask_boxes:

        cv2.rectangle(
            yolo_input,
            (
                mask_x1,
                mask_y1
            ),
            (
                mask_x2,
                mask_y2
            ),
            (
                0,
                0,
                0
            ),
            -1
        )


    # ========================================================
    # YOLO PERSON DETECTION
    # ========================================================

    yolo_detections = (
        run_person_detection(
            yolo_input
        )
    )


    # ========================================================
    # REMOVE HAND FALSE POSITIVES
    # ========================================================

    filtered_detections = []


    for detection in yolo_detections:

        center_x = (
            detection["x1"]
            + detection["x2"]
        ) // 2

        center_y = (
            detection["y1"]
            + detection["y2"]
        ) // 2

        inside_hand = False


        for (
            mask_x1,
            mask_y1,
            mask_x2,
            mask_y2
        ) in hand_mask_boxes:

            if (
                mask_x1
                <= center_x
                <= mask_x2

                and

                mask_y1
                <= center_y
                <= mask_y2
            ):

                inside_hand = True

                break


        if not inside_hand:

            filtered_detections.append(
                detection
            )


    yolo_detections = (
        filtered_detections
    )


    # ========================================================
    # DETECTION COUNT
    # ========================================================

    detected_person_count = len(
        yolo_detections
    )

    highest_person_confidence = max(
        (
            d["confidence"]
            for d in yolo_detections
        ),
        default=0.0
    )


    draw_person_detections(
        frame,
        yolo_detections
    )


    # ========================================================
    # SINGLE CLICK PER PINCH
    # ========================================================

    any_pinching = any(
        hand["pinching"]
        for hand in pinching_hands
    )

    pinch_started = (
        any_pinching
        and not pinch_was_active
    )


    if pinch_started:

        clicked = False


        for hand_data in pinching_hands:

            if (
                clicked
                or not hand_data["pinching"]
            ):

                continue


            hx = hand_data[
                "index_x"
            ]

            hy = hand_data[
                "index_y"
            ]


            for (
                panel_key,
                x,
                y,
                side
            ) in panel_definitions:

                button_x = (
                    x + PANEL_W
                    if side == "left"
                    else x
                )

                button_y = (
                    y + 20
                )

                distance = math.hypot(
                    hx - button_x,
                    hy - button_y
                )


                if distance <= 45:

                    panels[
                        panel_key
                    ] = not panels[
                        panel_key
                    ]

                    print(
                        f"{panel_key} -> "
                        f"{'OPEN' if panels[panel_key] else 'CLOSED'}"
                    )

                    clicked = True

                    break


            if clicked:
                break


    pinch_was_active = (
        any_pinching
    )


    # ========================================================
    # MR DIRECTIONAL MARKER
    # ========================================================

    person_detected_by_yolo = (
        len(yolo_detections) > 0
    )


    draw_mr_alignment(
        frame,

        direction_aligned
        and not person_detected_by_yolo,

        target_bearing,

        camera_heading,

        heading_error,

        target_distance,

        target_screen
    )

    if not person_detected_by_yolo:
        draw_gps_alignment_alert(
            frame,
            gps_alert
        )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "GODS EYE",
        (
            25,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        "SEARCH & RESCUE SYSTEM",
        (
            25,
            60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        WHITE,
        1
    )


    # ========================================================
    # DRAW SIX PANELS
    # ========================================================

    for (
        panel_key,
        x,
        y,
        side
    ) in panel_definitions:

        if panels[panel_key]:

            draw_panel(
                frame,
                x,
                y,
                PANEL_W,
                PANEL_H,
                panel_key.replace(
                    "_",
                    " "
                ).upper(),
                panel_key,
                side
            )

            draw_panel_content(
                frame,
                x,
                y,
                PANEL_W,
                PANEL_H,
                panel_key
            )

        else:

            button_x = (
                x + PANEL_W
                if side == "left"
                else x
            )

            button_y = (
                y + 20
            )

            hovered = False


            for hand_data in pinching_hands:

                distance = math.hypot(

                    hand_data[
                        "index_x"
                    ]
                    - button_x,

                    hand_data[
                        "index_y"
                    ]
                    - button_y
                )


                if distance <= 45:

                    hovered = True

                    break


            draw_control_button(
                frame,
                button_x,
                button_y,
                visible=False,
                hovered=hovered
            )


    # ========================================================
    # PHONE 2 SHARED FRAME
    # ========================================================

    SHARED_FRAME_PATH = (
        SCRIPT_DIR
        / "latest_frame.jpg"
    )

    cv2.imwrite(
        str(
            SHARED_FRAME_PATH
        ),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            88
        ]
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Gods Eye - Interactive UI",
        frame
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

detector.close()