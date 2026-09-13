import json
import math
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

GPS_DATA_PATH = SCRIPT_DIR / "gps_data.json"
GPS_ALERT_PATH = SCRIPT_DIR / "gps_alert.json"

# Target is considered aligned when the difference is 60 degrees or less.
ALIGNMENT_TOLERANCE_DEGREES = 60.0


def safe_float(value, default=None):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if value in ("", "--", "---", "N/A", "NA", "null", "None"):
                return default

        return float(value)

    except (ValueError, TypeError):
        return default


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(delta_lon)
    )

    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360.0) % 360.0


def calculate_distance_meters(lat1, lon1, lat2, lon2):
    earth_radius = 6371000.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(max(0.0, 1.0 - a))
    )

    return earth_radius * c


def angular_difference(a, b):
    """Return the smallest absolute difference between two headings."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def read_gps_data():
    try:
        with GPS_DATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError
    ):
        return None


def write_alert(result):
    # Write atomically so main.py never reads a half-written JSON file.
    temp_path = GPS_ALERT_PATH.with_suffix(".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=4)

        temp_path.replace(GPS_ALERT_PATH)

    except OSError:
        pass


def main():
    print("=" * 60)
    print("GODS EYE - GPS COMPARISON")
    print("=" * 60)
    print("Reading gps_data.json...")
    print(f"Alignment threshold: <= {ALIGNMENT_TOLERANCE_DEGREES:.0f}°")
    print()

    while True:
        data = read_gps_data()

        if not data:
            print("\rWaiting for GPS data...", end="", flush=True)
            time.sleep(0.2)
            continue

        soldier = data.get("soldier", {})
        person = data.get("person", {})

        # --------------------------------------------------
        # SOLDIER
        # --------------------------------------------------

        soldier_lat = safe_float(soldier.get("latitude"))
        soldier_lon = safe_float(soldier.get("longitude"))
        soldier_accuracy = safe_float(soldier.get("accuracy"))
        soldier_heading = safe_float(soldier.get("heading"))
        soldier_received = bool(soldier.get("received", False))

        # --------------------------------------------------
        # TARGET / PERSON
        # --------------------------------------------------

        person_lat = safe_float(person.get("latitude"))
        person_lon = safe_float(person.get("longitude"))
        person_accuracy = safe_float(person.get("accuracy"))
        person_heading = safe_float(person.get("heading"))
        person_received = bool(person.get("received", False))

        gps_valid = (
            soldier_received
            and person_received
            and soldier_lat is not None
            and soldier_lon is not None
            and person_lat is not None
            and person_lon is not None
            and soldier_heading is not None
        )

        if not gps_valid:
            result = {
                "gps_valid": False,
                "aligned": False,
                "soldier": {
                    "latitude": soldier_lat,
                    "longitude": soldier_lon,
                    "accuracy": soldier_accuracy,
                    "heading": soldier_heading,
                    "received": soldier_received
                },
                "target": {
                    "latitude": person_lat,
                    "longitude": person_lon,
                    "accuracy": person_accuracy,
                    "heading": person_heading,
                    "received": person_received
                },
                "bearing": None,
                "difference": None,
                "distance": None
            }

            write_alert(result)

            print("\rWaiting for valid Soldier + Target GPS data...",
                  end="", flush=True)

            time.sleep(0.2)
            continue

        # --------------------------------------------------
        # TARGET BEARING FROM SOLDIER
        # --------------------------------------------------

        target_bearing = calculate_bearing(
            soldier_lat,
            soldier_lon,
            person_lat,
            person_lon
        )

        # --------------------------------------------------
        # DISTANCE
        # --------------------------------------------------

        distance = calculate_distance_meters(
            soldier_lat,
            soldier_lon,
            person_lat,
            person_lon
        )

        # --------------------------------------------------
        # HEADING DIFFERENCE
        # --------------------------------------------------

        heading_difference = angular_difference(
            soldier_heading,
            target_bearing
        )

        # --------------------------------------------------
        # ALIGNMENT
        # <= 60 DEGREES = ALIGNED
        # --------------------------------------------------

        aligned = (
            heading_difference <= ALIGNMENT_TOLERANCE_DEGREES
        )

        # --------------------------------------------------
        # DATA FOR HUD
        # --------------------------------------------------

        result = {
            "gps_valid": True,
            "aligned": aligned,

            "soldier": {
                "latitude": soldier_lat,
                "longitude": soldier_lon,
                "accuracy": soldier_accuracy,
                "heading": soldier_heading,
                "received": soldier_received
            },

            "target": {
                "latitude": person_lat,
                "longitude": person_lon,
                "accuracy": person_accuracy,
                "heading": person_heading,
                "received": person_received
            },

            "bearing": target_bearing,
            "difference": heading_difference,
            "distance": distance
        }

        write_alert(result)

        # --------------------------------------------------
        # TERMINAL OUTPUT
        # --------------------------------------------------

        print("\033[H\033[J", end="")

        print("=" * 60)
        print("GODS EYE - GPS COMPARISON")
        print("=" * 60)

        print()
        print("SOLDIER")
        print("-" * 30)
        print(f"Latitude  : {soldier_lat:.6f}")
        print(f"Longitude : {soldier_lon:.6f}")
        print(f"Accuracy  : {soldier_accuracy} m")
        print(f"Heading   : {soldier_heading:.2f}°")
        print(f"Received  : {soldier_received}")

        print()
        print("TARGET")
        print("-" * 30)
        print(f"Latitude  : {person_lat:.6f}")
        print(f"Longitude : {person_lon:.6f}")
        print(f"Accuracy  : {person_accuracy} m")
        print(
            f"Heading   : "
            f"{person_heading:.2f}°"
            if person_heading is not None
            else "Heading   : None"
        )
        print(f"Received  : {person_received}")

        print()
        print("CALCULATION")
        print("-" * 30)
        print(f"Target Bearing : {target_bearing:.2f}°")
        print(f"Soldier Heading: {soldier_heading:.2f}°")
        print(f"Difference     : {heading_difference:.2f}°")
        print(f"Distance       : {distance:.2f} m")

        print()
        print(
            ">>> TARGET ALIGNED <<<"
            if aligned
            else ">>> TARGET NOT ALIGNED <<<"
        )

        print()
        print(
            f"Alignment rule: "
            f"difference <= {ALIGNMENT_TOLERANCE_DEGREES:.0f}°"
        )

        time.sleep(0.2)


if __name__ == "__main__":
    main()
