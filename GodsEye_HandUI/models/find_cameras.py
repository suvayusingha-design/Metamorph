import cv2

print("Checking cameras...\n")

for i in range(10):

    camera = cv2.VideoCapture(i)

    if camera.isOpened():
        print(f"Camera {i} FOUND")
        camera.release()
    else:
        print(f"Camera {i} not available")