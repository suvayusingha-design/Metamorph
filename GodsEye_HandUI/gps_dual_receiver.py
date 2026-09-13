import socket
import threading
import json
import os
from pathlib import Path

HOST = "0.0.0.0"
SOLDIER_PORT = 5000
PERSON_PORT = 5001

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "gps_data.json"
DATA_LOCK = threading.Lock()

latest_data = {
    "soldier": {
        "latitude": 0.0,
        "longitude": 0.0,
        "accuracy": 0.0,
        "heading": 0.0,
        "received": False
    },
    "person": {
        "latitude": 0.0,
        "longitude": 0.0,
        "accuracy": 0.0,
        "received": False
    }
}


def safe_float(value, default=0.0):
    """Safely parse Android sensor values such as --, null or NaN."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() in ("", "--", "null", "None", "nan"):
            return default
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def save_data():
    """Write the current GPS data to gps_data.json."""
    temp_path = DATA_PATH.with_suffix(".tmp")

    # IMPORTANT:
    # Do not acquire DATA_LOCK here.
    # update_soldier/update_person already hold it.
    payload = json.dumps(latest_data, indent=2)

    try:
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, DATA_PATH)
    except PermissionError as e:
        print(f"\n[FILE ERROR] Cannot update {DATA_PATH}: {e}")


def update_soldier(sensor):
    with DATA_LOCK:
        previous = latest_data["soldier"]
        latest_data["soldier"] = {
            "latitude": safe_float(sensor.get("latitude"), previous["latitude"]),
            "longitude": safe_float(sensor.get("longitude"), previous["longitude"]),
            "accuracy": safe_float(sensor.get("accuracy"), previous["accuracy"]),
            "heading": safe_float(sensor.get("heading"), previous["heading"]),
            "received": True
        }

        # Save while the data lock is held. save_data() does not acquire it.
        save_data()


def update_person(sensor):
    with DATA_LOCK:
        previous = latest_data["person"]
        latest_data["person"] = {
            "latitude": safe_float(sensor.get("latitude"), previous["latitude"]),
            "longitude": safe_float(sensor.get("longitude"), previous["longitude"]),
            "accuracy": safe_float(sensor.get("accuracy"), previous["accuracy"]),
            "received": True
        }

        save_data()

def handle_client(conn, addr, role):
    print(f"\n[{role}] CONNECTED: {addr[0]}:{addr[1]}")

    buffer = ""

    try:
        while True:
            data = conn.recv(4096)

            if not data:
                print(f"\n[{role}] DISCONNECTED")
                break

            buffer += data.decode("utf-8", errors="replace")

            while "\n" in buffer:
                message, buffer = buffer.split("\n", 1)
                message = message.strip()

                if not message:
                    continue

                try:
                    sensor = json.loads(message)

                    if role == "SOLDIER":
                        update_soldier(sensor)

                        print(
                            f"\r[SOLDIER] "
                            f"Lat: {safe_float(sensor.get('latitude', 0)):.6f} | "
                            f"Lon: {safe_float(sensor.get('longitude', 0)):.6f} | "
                            f"Acc: {safe_float(sensor.get('accuracy', 0)):.1f} m | "
                            f"Heading: {safe_float(sensor.get('heading', 0)):.1f}°    ",
                            end="",
                            flush=True
                        )

                    else:
                        update_person(sensor)

                        print(
                            f"\r[TEST PERSON] "
                            f"Lat: {safe_float(sensor.get('latitude', 0)):.6f} | "
                            f"Lon: {safe_float(sensor.get('longitude', 0)):.6f} | "
                            f"Acc: {safe_float(sensor.get('accuracy', 0)):.1f} m    ",
                            end="",
                            flush=True
                        )

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"\n[{role}] Invalid data: {e}")

    except ConnectionResetError:
        print(f"\n[{role}] Connection reset by phone.")

    except Exception as e:
        print(f"\n[{role}] Network error: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass


def start_server(port, role):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(5)

    print(f"[{role}] Listening on 0.0.0.0:{port}")

    while True:
        conn, addr = server.accept()

        threading.Thread(
            target=handle_client,
            args=(conn, addr, role),
            daemon=True
        ).start()


def main():
    print("=" * 60)
    print("GOD'S EYE - GPS RECEIVER + MAIN.PY DATA BRIDGE")
    print("=" * 60)
    print("Laptop IP: 192.168.0.102")
    print(f"Soldier receiver: 192.168.0.102:{SOLDIER_PORT}")
    print(f"Person receiver : 192.168.0.102:{PERSON_PORT}")
    print(f"Data bridge     : {DATA_PATH}")
    print("=" * 60)

    # Initial file.
    save_data()

    threading.Thread(
        target=start_server,
        args=(SOLDIER_PORT, "SOLDIER"),
        daemon=True
    ).start()

    threading.Thread(
        target=start_server,
        args=(PERSON_PORT, "TEST PERSON"),
        daemon=True
    ).start()

    threading.Event().wait()


if __name__ == "__main__":
    main()
