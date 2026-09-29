import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

image_files = ["polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg"]

# Parameters for SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 250

params.filterByArea = True
params.minArea = 20
params.maxArea = 20000

params.filterByColor = True
params.blobColor = 0

params.filterByCircularity = True
params.minCircularity = 0.5

detector = cv2.SimpleBlobDetector_create(params)

for filename in image_files:
    image_path = os.path.join(script_dir, "images", filename)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print(f"Could not load {filename}")
        continue

    keypoints = detector.detect(img_gray)

    print(f"{filename}: found {len(keypoints)} blobs")
    for kp in keypoints:
        x, y = kp.pt
        print(f"  blob at ({x:.1f}, {y:.1f}), size={kp.size:.1f}")

    output = cv2.drawKeypoints(
        img_gray, keypoints, np.array([]), (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imshow(f"Blobs - {filename}", output)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Challenge 1: Contour filtering

for filename in image_files:
    image_path = os.path.join(script_dir, "images", filename)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        continue

    
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = 20
    max_area = 20000
    filtered = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    print(f"{filename} (contours): found {len(filtered)} blobs after filtering")

    output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for c in filtered:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            print(f"  blob center: ({cx}, {cy})")

    cv2.imshow(f"Contours - {filename}", output)

cv2.waitKey(0)
cv2.destroyAllWindows()