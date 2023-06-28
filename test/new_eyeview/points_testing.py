import cv2
from src.config import *
import imutils

global image
mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (70, 70, 70),
                     2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))


cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
capture = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = capture.read()
    input_size = 416
    original_image = imutils.resize(frame, width=500)
    frame = original_image
    # original_image=frame
    (H, W) = original_image.shape[:2]
    while True:
        image = original_image
        cv2.imshow("image", image)
        cv2.waitKey(1)
        if len(mouse_pts) == 8:
            cv2.destroyWindow("image")
            break

    points = mouse_pts
    POINTS=points
    print(points)
    break
capture.release()
