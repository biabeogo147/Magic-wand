import os
import cv2
import util
import torch
import argparse
import numpy as np
import magic_wand_model
from collections import deque


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-s", "--canvas", type=bool, default=True, help="Display black & white canvas")
    parser.add_argument("-m", "--model_path", type=str, default="model", help="Model path")
    args = parser.parse_args()
    return args


def main(args):
    color_lower = np.array(util.BLUE_HSV_LOWER)
    color_upper = np.array(util.BLUE_HSV_UPPER)
    color_pointer = util.BLUE_RGB

    # Initialize deque for storing detected points and canvas for drawing
    frame_width, frame_height = 640, 480
    points = deque(maxlen=512)
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Load the video from camera (Here I use built-in webcam)
    camera = cv2.VideoCapture(0)
    predicted_class = None
    is_drawing = False
    is_show = False

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = magic_wand_model.magic_wand_model(num_classes=len(util.classes))
    if os.path.exists(os.path.join(args.model_path, "best.pt")):
        checkpoint = torch.load(os.path.join(args.model_path, "best.pt"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        # Detect pixels fall within the pre-defined color range. Then, blur the image
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_show = False
            is_drawing = not is_drawing
            if is_drawing:
                points = deque(maxlen=512)
                canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            else:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    x, y, w, h = cv2.boundingRect(contours[0])

                    # (ROI - Region of Interest)
                    roi = canvas[y:y + h, x:x + w]
                    roi = (255 - cv2.resize(roi, (28, 28))) / 255.0
                    roi = torch.tensor(roi, dtype=torch.float32).to(device)

                    output = model(roi.unsqueeze(0).unsqueeze(0))
                    predicted_class = util.classes[torch.argmax(output).item()]
                    is_show = True

        if is_show:
            cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, util.WHITE_RGB, 2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            # Take the biggest contour, since it is possible that there are other objects in front of camera
            # whose color falls within the range of our pre-defined color
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), util.GREEN_RGB, 2)
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], util.WHITE_RGB, 5)
                    cv2.line(frame, points[i - 1], points[i], color_pointer, 2)

        cv2.imshow("Camera", frame)
        if args.canvas:
            cv2.imshow("Canvas", 255-canvas)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    main(args)
