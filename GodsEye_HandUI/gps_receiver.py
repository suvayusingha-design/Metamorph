import socket
import json
import math

HOST = "0.0.0.0"
PORT = 5000


def heading_to_direction(heading):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    index = int((heading + 11.25) / 22.5) % 16
    return directions[index]


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(1)

print(f"Waiting for soldier phone on port {PORT}...")

conn, addr = server.accept()

print("Phone connected:", addr)

buffer = ""

while True:
    data = conn.recv(4096)

    if not data:
        break

    buffer += data.decode("utf-8")

    while "\n" in buffer:
        message, buffer = buffer.split("\n", 1)

        try:
            sensor_data = json.loads(message)

            latitude = sensor_data["latitude"]
            longitude = sensor_data["longitude"]
            accuracy = sensor_data["accuracy"]
            heading = sensor_data["heading"]

            direction = heading_to_direction(heading)

            print(
                f"\r"
                f"Lat: {latitude:.6f} | "
                f"Lon: {longitude:.6f} | "
                f"Accuracy: {accuracy:.1f} m | "
                f"Heading: {heading:6.1f}° | "
                f"Direction: {direction:3s}",
                end="",
                flush=True
            )

        except json.JSONDecodeError:
            print("\nInvalid data:", message)

        except KeyError as e:
            print(f"\nMissing field: {e}")


conn.close()
server.close()