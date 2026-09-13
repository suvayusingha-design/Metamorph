from flask import Flask, Response
from pathlib import Path
import cv2
import numpy as np
import time
import socket
import threading

# ============================================================
# GODS EYE — PHONE 2 VR STREAM
# ============================================================
#
# This is a SEPARATE program from main.py.
#
# main.py writes:
#     latest_frame.jpg
#
# This program:
#     1. reads that processed frame
#     2. serves it to Phone 2
#     3. Phone 2 displays two lens-like eye views
#     4. keeps a black VR-style background
#
# Phone 2:
#     http://LAPTOP_IP:5000
#
# ============================================================

app = Flask(__name__)

FRAME_PATH = Path("latest_frame.jpg")

STREAM_FPS = 20
JPEG_QUALITY = 88

# Base eye size. CSS scales this on the actual phone screen.
EYE_WIDTH = 600
EYE_HEIGHT = 338

latest_frame_jpeg = None
frame_lock = threading.Lock()
running = True


def get_local_ip():
    """Return the laptop's local Wi-Fi IP address."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()

    return ip


def frame_worker():
    """Read the processed Gods Eye frame and encode it for streaming."""

    global latest_frame_jpeg

    last_mtime = -1

    while running:

        try:
            if not FRAME_PATH.exists():
                time.sleep(0.05)
                continue

            mtime = FRAME_PATH.stat().st_mtime_ns

            if mtime == last_mtime:
                time.sleep(1 / STREAM_FPS)
                continue

            last_mtime = mtime

            frame = cv2.imread(
                str(FRAME_PATH),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                time.sleep(0.02)
                continue

            # Keep the complete Gods Eye frame.
            # No crop and no stretch here.
            frame = cv2.resize(
                frame,
                (EYE_WIDTH, EYE_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            if ok:
                with frame_lock:
                    latest_frame_jpeg = encoded.tobytes()

        except Exception as exc:
            print("Frame worker error:", exc)
            time.sleep(0.1)


def mjpeg_generator():

    while True:

        with frame_lock:
            data = latest_frame_jpeg

        if data is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store\r\n\r\n"
                + data +
                b"\r\n"
            )

        time.sleep(1 / STREAM_FPS)


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0,
               maximum-scale=1.0,
               user-scalable=no,
               viewport-fit=cover">

<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">

<title>Gods Eye VR</title>

<style>

* {
    box-sizing: border-box;
}

html, body {
    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    background: #000;

    overflow: hidden;

    touch-action: none;
}

body {
    display: flex;
    align-items: center;
    justify-content: center;

    background: #000;
}

/* =========================================================
   VR STAGE
   ========================================================= */

#vrStage {
    position: relative;

    width: 100vw;
    height: 100vh;

    background: #000;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: hidden;
}

/* =========================================================
   EYE CONTAINER
   ========================================================= */

#eyes {
    display: flex;

    align-items: center;
    justify-content: center;

    gap: clamp(12px, 2vw, 34px);

    width: 100%;
}

/* =========================================================
   EACH VR EYE
   ========================================================= */

.eye {
    position: relative;

    /*
       This is intentionally smaller than half the display.
       That leaves black space around the lens image, similar
       to the VR presentation style you referenced.
    */

    width: min(36vw, 620px);
    aspect-ratio: 16 / 9;

    background: #000;

    overflow: hidden;

    /*
       Soft lens-like curved edges.
    */
    border-radius: 13% 13% 15% 15% / 11% 11% 12% 12%;

    box-shadow:
        0 0 0 1px rgba(255,255,255,0.05),
        0 8px 35px rgba(0,0,0,0.8);
}

.eye img {
    width: 100%;
    height: 100%;

    display: block;

    object-fit: contain;

    background: #000;

    user-select: none;
    -webkit-user-select: none;
    -webkit-touch-callout: none;
    pointer-events: none;
}

/* =========================================================
   CENTER DIVIDER
   ========================================================= */

#divider {
    position: absolute;

    left: 50%;
    top: 35%;
    transform: translateX(-50%);

    width: 2px;
    height: 30%;

    background: rgba(255,255,255,0.70);

    pointer-events: none;
}

/* =========================================================
   YOUTUBE-VR-LIKE TOP CONTROLS
   ========================================================= */

.control {
    position: fixed;

    top: max(18px, env(safe-area-inset-top));

    z-index: 20;

    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    color: white;

    background: transparent;

    border: 0;

    font-size: 34px;
    line-height: 1;

    opacity: 0.95;

    cursor: pointer;
}

#closeBtn {
    left: max(18px, env(safe-area-inset-left));
}

#settingsBtn {
    right: max(18px, env(safe-area-inset-right));

    font-size: 28px;
}

/* =========================================================
   FULLSCREEN BUTTON
   ========================================================= */

#fullscreenBtn {
    position: fixed;

    right: 20px;
    bottom: max(20px, env(safe-area-inset-bottom));

    z-index: 30;

    padding: 10px 14px;

    border: 1px solid rgba(255,255,255,0.55);
    border-radius: 8px;

    color: white;

    background: rgba(0,0,0,0.65);

    font-size: 13px;

    backdrop-filter: blur(8px);

    cursor: pointer;
}

/* Hide browser controls button once fullscreen begins. */
:fullscreen #fullscreenBtn,
:-webkit-full-screen #fullscreenBtn {
    display: none;
}

/* =========================================================
   FULLSCREEN
   ========================================================= */

:fullscreen #vrStage,
:-webkit-full-screen #vrStage {
    width: 100vw;
    height: 100vh;
}

/* =========================================================
   SMALLER PORTRAIT/UNSUPPORTED FALLBACK
   ========================================================= */

@media (orientation: portrait) {

    .eye {
        width: 44vw;
    }

    #eyes {
        gap: 2vw;
    }

    #divider {
        height: 28%;
        top: 36%;
    }
}

</style>

</head>

<body>

<div id="vrStage">

    <div id="eyes">

        <div class="eye">
            <img src="/eye_stream" draggable="false">
        </div>

        <div class="eye">
            <img src="/eye_stream" draggable="false">
        </div>

    </div>

    <div id="divider"></div>

</div>

<button id="closeBtn"
        class="control"
        aria-label="Exit fullscreen">
    ×
</button>

<button id="settingsBtn"
        class="control"
        aria-label="Settings">
    ⚙
</button>

<button id="fullscreenBtn"
        onclick="goFullscreen()">
    FULL SCREEN
</button>

<script>

async function goFullscreen() {

    try {

        const stage = document.documentElement;

        if (stage.requestFullscreen) {

            await stage.requestFullscreen();

        } else if (stage.webkitRequestFullscreen) {

            stage.webkitRequestFullscreen();

        }

        /*
         * Request landscape orientation for the headset.
         * Some mobile browsers do not allow this from a normal tab.
         */

        if (
            screen.orientation &&
            screen.orientation.lock
        ) {

            try {

                await screen.orientation.lock("landscape");

            } catch (e) {

                console.log(
                    "Landscape lock not available."
                );
            }
        }

    } catch (error) {

        console.log(
            "Fullscreen unavailable:",
            error
        );
    }
}


document
    .getElementById("closeBtn")
    .addEventListener("click", async function () {

        try {

            if (document.fullscreenElement) {

                await document.exitFullscreen();

            } else if (document.webkitFullscreenElement) {

                document.webkitExitFullscreen();

            }

        } catch (e) {

            console.log(e);

        }

    });


document
    .getElementById("settingsBtn")
    .addEventListener("click", function () {

        /*
         * Placeholder for later:
         * IPD / eye size / brightness / alignment.
         */

        console.log(
            "VR settings will be added later."
        );

    });


/*
 * Request landscape automatically after user interaction,
 * where supported.
 */

document.addEventListener(
    "click",
    async function () {

        if (
            screen.orientation &&
            screen.orientation.lock
        ) {

            try {

                await screen.orientation.lock(
                    "landscape"
                );

            } catch (e) {}

        }

    },
    { once: true }
);

</script>

</body>
</html>
"""


@app.route("/eye_stream")
def eye_stream():

    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    worker = threading.Thread(
        target=frame_worker,
        daemon=True
    )

    worker.start()

    ip = get_local_ip()

    print("")
    print("============================================================")
    print("             GODS EYE — VR DISPLAY")
    print("============================================================")
    print("")
    print("Make sure main.py is running.")
    print("")
    print("Open Phone 2 in LANDSCAPE and visit:")
    print("")
    print(f"        http://{ip}:5000")
    print("")
    print("Press FULL SCREEN on the phone.")
    print("")
    print("The display uses two smaller lens-style eye views")
    print("on a black background.")
    print("")
    print("Press Ctrl+C here to stop the VR server.")
    print("============================================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )
