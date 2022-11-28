# This program is used for capturing chessboard images.
# The device number and image saved path may need to be changed.
# Press 's' to save an image and press 'q' to quit program.
 
import numpy as np
import cv2

# open device
cap = cv2.VideoCapture(0)  # The device number may need to be changed.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# set image saved path
image_save_path = "calibration_image/" # The image saved path may need to be changed.
# image_save_path = "test_calibration/" # The image saved path may need to be changed.

# set window size
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("frame", 800, 800)

count = 0

while(True):
    ret, frame = cap.read()

    cv2.imshow("frame", frame)
    k = cv2.waitKey(10) & 0xFF

    # save frame
    if k == ord('s'):   # press 's' to save an image
        cv2.imwrite(image_save_path + str(count) + ".png", frame)
        print(count)
        count += 1

    # quit program
    if k == ord('q'):  # press 'q' to quit program
        break

cap.release()
cv2.destroyAllWindows()
