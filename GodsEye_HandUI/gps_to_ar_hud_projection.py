"""
GPS -> AR HUD target projection
--------------------------------
Projects a geospatial target into a camera/HUD frame.

Assumptions:
- Coordinates are WGS84 latitude/longitude in decimal degrees.
- Heading is degrees clockwise from TRUE North.
- Terrain is flat and target/observer elevation difference is 0.
- Camera optical axis is aligned with the observer's heading after applying
  CAMERA_HEADING_OFFSET_DEG.
- The wall is an occluder only; it does not change the target's bearing.
- Horizontal camera FOV is known/estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_HORIZONTAL_FOV_DEG = 70.0

# Use this if the phone's compass axis and camera optical axis differ.
# Example: if the phone reports 90 degrees when the camera points north,
# calibrate this value accordingly.
CAMERA_HEADING_OFFSET_DEG = 0.0

# Set to -1.0 if the target moves in the wrong left/right direction.
CAMERA_X_DIRECTION = 1.0

# Current experiment: flat terrain / zero elevation difference.
TARGET_ELEVATION_DEG = 0.0

# Bounding-box depth scaling.
# This does NOT calculate a real person's angular size unless the person's
# physical dimensions are known. It provides a stable HUD visualization.
REFERENCE_DISTANCE_M = 20.0
REFERENCE_BOX_WIDTH_PX = 150.0
REFERENCE_BOX_HEIGHT_PX = 105.0

MIN_BOX_SCALE = 0.60
MAX_BOX_SCALE = 1.60

MIN_BOX_WIDTH_PX = 70
MAX_BOX_WIDTH_PX = 240
MIN_BOX_HEIGHT_PX = 50
MAX_BOX_HEIGHT_PX = 170

# Keep the complete box inside the frame.
EDGE_MARGIN_PX = 8


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GPSPoint:
    latitude: float
    longitude: float


@dataclass
class ScreenTarget:
    visible: bool
    x: int | None
    y: int | None
    width: int | None
    height: int | None

    distance_m: float
    bearing_deg: float
    camera_heading_deg: float
    delta_deg: float

    reason: str


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------

def normalize_heading(degrees: float) -> float:
    """Normalize any heading into [0, 360)."""
    return degrees % 360.0


def validate_gps(point: GPSPoint) -> None:
    """Raise ValueError for invalid WGS84 coordinates."""
    if not -90.0 <= point.latitude <= 90.0:
        raise ValueError(f"Invalid latitude: {point.latitude}")

    if not -180.0 <= point.longitude <= 180.0:
        raise ValueError(f"Invalid longitude: {point.longitude}")


# ---------------------------------------------------------------------------
# 1. Forward azimuth / initial bearing
# ---------------------------------------------------------------------------

def initial_bearing(observer: GPSPoint, target: GPSPoint) -> float:
    """
    Calculate the initial forward azimuth from observer -> target.

    Result:
        0°   = North
        90°  = East
        180° = South
        270° = West
    """
    validate_gps(observer)
    validate_gps(target)

    lat1 = math.radians(observer.latitude)
    lat2 = math.radians(target.latitude)
    delta_lon = math.radians(
        target.longitude - observer.longitude
    )

    y = math.sin(delta_lon) * math.cos(lat2)

    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(delta_lon)
    )

    bearing = math.degrees(math.atan2(y, x))

    return normalize_heading(bearing)


# ---------------------------------------------------------------------------
# 2. Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance_m(observer: GPSPoint, target: GPSPoint) -> float:
    """Return great-circle surface distance in metres."""
    validate_gps(observer)
    validate_gps(target)

    earth_radius_m = 6_371_000.0

    lat1 = math.radians(observer.latitude)
    lat2 = math.radians(target.latitude)

    dlat = lat2 - lat1
    dlon = math.radians(
        target.longitude - observer.longitude
    )

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    # Clamp against tiny floating-point overshoots.
    a = max(0.0, min(1.0, a))

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(1.0 - a)
    )

    return earth_radius_m * c


# ---------------------------------------------------------------------------
# 3. Signed angular difference
# ---------------------------------------------------------------------------

def signed_angle_difference(
    target_bearing_deg: float,
    camera_heading_deg: float,
) -> float:
    """
    Return target bearing relative to camera heading in [-180, +180).

    Convention used here:
        negative -> target is LEFT of camera center
        positive -> target is RIGHT of camera center
        zero     -> target is straight ahead

    The modulo operation handles the 0°/360° wrap-around.

    Example:
        heading = 359°
        target  = 1°
        result  = +2°, NOT -358°

        heading = 1°
        target  = 359°
        result  = -2°, NOT +358°
    """
    return (
        target_bearing_deg
        - camera_heading_deg
        + 180.0
    ) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 4. Distance-based box scaling
# ---------------------------------------------------------------------------

def calculate_box_size(
    distance_m: float,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """
    Scale the HUD box inversely with distance.

    A target at REFERENCE_DISTANCE_M gets the reference box size.
    Closer target -> larger box.
    Farther target -> smaller box.

    This is a visualization/depth cue. For physically correct target
    angular size, the real target height/width and camera intrinsics should
    also be supplied.
    """
    distance_m = max(distance_m, 0.1)

    scale = REFERENCE_DISTANCE_M / distance_m

    scale = max(
        MIN_BOX_SCALE,
        min(MAX_BOX_SCALE, scale),
    )

    width = int(REFERENCE_BOX_WIDTH_PX * scale)
    height = int(REFERENCE_BOX_HEIGHT_PX * scale)

    # Keep the box sensible for the actual display resolution.
    width = max(
        MIN_BOX_WIDTH_PX,
        min(MAX_BOX_WIDTH_PX, width),
    )

    height = max(
        MIN_BOX_HEIGHT_PX,
        min(MAX_BOX_HEIGHT_PX, height),
    )

    width = min(width, max(20, screen_width - 2 * EDGE_MARGIN_PX))
    height = min(height, max(20, screen_height - 2 * EDGE_MARGIN_PX))

    return width, height


# ---------------------------------------------------------------------------
# 5. Horizontal angle -> screen X
# ---------------------------------------------------------------------------

def project_horizontal_angle_to_x(
    delta_deg: float,
    fov_h_deg: float,
    screen_width: int,
) -> int | None:
    """
    Project a horizontal viewing angle into pixel X using a pinhole-camera
    model.

    At:
        delta = -FOV/2 -> left edge
        delta = 0       -> screen center
        delta = +FOV/2 -> right edge

    A tangent projection is used instead of a simple linear mapping.
    """
    if fov_h_deg <= 0.0 or fov_h_deg >= 180.0:
        raise ValueError("FOV_h must be between 0 and 180 degrees.")

    half_fov = fov_h_deg / 2.0

    if abs(delta_deg) > half_fov:
        return None

    half_width = screen_width / 2.0

    focal_length_px = (
        half_width
        / math.tan(math.radians(half_fov))
    )

    x = (
        half_width
        + CAMERA_X_DIRECTION
        * focal_length_px
        * math.tan(math.radians(delta_deg))
    )

    return int(round(x))


# ---------------------------------------------------------------------------
# 6. Complete GPS target -> HUD projection
# ---------------------------------------------------------------------------

def project_target(
    observer: GPSPoint,
    target: GPSPoint,
    observer_heading_deg: float,
    fov_h_deg: float,
    screen_width: int,
    screen_height: int,
) -> ScreenTarget:
    """
    Complete geospatial-to-HUD calculation.

    Returns all useful intermediate values plus the final screen position.
    """

    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("Screen dimensions must be positive.")

    target_bearing = initial_bearing(
        observer,
        target,
    )

    distance_m = haversine_distance_m(
        observer,
        target,
    )

    camera_heading = normalize_heading(
        observer_heading_deg
        + CAMERA_HEADING_OFFSET_DEG
    )

    delta_deg = signed_angle_difference(
        target_bearing,
        camera_heading,
    )

    half_fov = fov_h_deg / 2.0

    # Visibility is purely angular here.
    # The wall does not make the GPS target "invisible" to this projection;
    # it is precisely what the AR marker is intended to represent.
    if abs(delta_deg) > half_fov:
        return ScreenTarget(
            visible=False,
            x=None,
            y=None,
            width=None,
            height=None,
            distance_m=distance_m,
            bearing_deg=target_bearing,
            camera_heading_deg=camera_heading,
            delta_deg=delta_deg,
            reason="TARGET_OUTSIDE_HORIZONTAL_FOV",
        )

    x = project_horizontal_angle_to_x(
        delta_deg,
        fov_h_deg,
        screen_width,
    )

    if x is None:
        return ScreenTarget(
            visible=False,
            x=None,
            y=None,
            width=None,
            height=None,
            distance_m=distance_m,
            bearing_deg=target_bearing,
            camera_heading_deg=camera_heading,
            delta_deg=delta_deg,
            reason="TARGET_OUTSIDE_HORIZONTAL_FOV",
        )

    box_w, box_h = calculate_box_size(
        distance_m,
        screen_width,
        screen_height,
    )

    # Flat terrain + 0 elevation difference:
    # put the target on the horizontal optical axis.
    y = screen_height // 2

    # Prevent the box itself from crossing the frame edge.
    min_x = EDGE_MARGIN_PX + box_w // 2
    max_x = screen_width - EDGE_MARGIN_PX - box_w // 2

    x = max(min_x, min(max_x, x))

    return ScreenTarget(
        visible=True,
        x=x,
        y=y,
        width=box_w,
        height=box_h,
        distance_m=distance_m,
        bearing_deg=target_bearing,
        camera_heading_deg=camera_heading,
        delta_deg=delta_deg,
        reason="TARGET_IN_FOV",
    )


# ---------------------------------------------------------------------------
# 7. OpenCV drawing helper
# ---------------------------------------------------------------------------

def draw_gps_target_box(
    frame,
    projection: ScreenTarget,
    color=(0, 255, 0),
) -> None:
    """
    Draw a green corner-style GPS target box.

    Works directly with a BGR OpenCV frame.
    """
    if not projection.visible:
        return

    h, w = frame.shape[:2]

    cx = projection.x
    cy = projection.y
    box_w = projection.width
    box_h = projection.height

    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w - 1, cx + box_w // 2)
    y2 = min(h - 1, cy + box_h // 2)

    corner = max(12, min(24, box_w // 6))

    # Four corner brackets.
    cv2 = __import__("cv2")

    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 3)

    cv2.line(frame, (x2 - corner, y1), (x2, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 3)

    cv2.line(frame, (x1, y2 - corner), (x1, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 3)

    cv2.line(frame, (x2 - corner, y2), (x2, y2), color, 3)
    cv2.line(frame, (x2, y2 - corner), (x2, y2), color, 3)

    # Center marker.
    cv2.drawMarker(
        frame,
        (cx, cy),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
    )

    label = "GPS TARGET"

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace these with live values from your two phones.
    soldier = GPSPoint(
        latitude=22.694748,
        longitude=88.379072,
    )

    person = GPSPoint(
        latitude=22.694583,
        longitude=88.379093,
    )

    soldier_heading = 173.0

    projection = project_target(
        observer=soldier,
        target=person,
        observer_heading_deg=soldier_heading,
        fov_h_deg=CAMERA_HORIZONTAL_FOV_DEG,
        screen_width=1920,
        screen_height=1080,
    )

    print(f"Bearing        : {projection.bearing_deg:.2f}°")
    print(f"Camera heading : {projection.camera_heading_deg:.2f}°")
    print(f"Delta          : {projection.delta_deg:+.2f}°")
    print(f"Distance       : {projection.distance_m:.2f} m")
    print(f"Visible        : {projection.visible}")
    print(f"Screen         : ({projection.x}, {projection.y})")
    print(f"Box            : {projection.width} x {projection.height}")
    print(f"Reason         : {projection.reason}")
