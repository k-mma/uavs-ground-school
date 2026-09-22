import cv2
import numpy as np

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "images", "tennis.jpg")

img = cv2.imread(image_path)
print("image shape:", img.shape)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# print(hsv[300, 300])
# print(hsv[1900, 900])

# maroon background
lower_bg = np.array([160, 80, 100])
upper_bg = np.array([179, 255, 255])
mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
background_only = cv2.bitwise_and(img, img, mask=mask_bg)

# gray frame
lower_frame = np.array([0, 0, 150])
upper_frame = np.array([179, 40, 255])
mask_frame = cv2.inRange(hsv, lower_frame, upper_frame)
frame_only = cv2.bitwise_and(img, img, mask=mask_frame)

cv2.imshow("background", background_only)
cv2.imshow("frame", frame_only)
cv2.waitKey(0)
cv2.destroyAllWindows()