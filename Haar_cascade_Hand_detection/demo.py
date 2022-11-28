import cv2
import numpy as np

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))

HAND_DETECTION_MODEL_PATH = [
    "./rpalm.xml",
    "./lpalm.xml",
    "./left.xml", # left rotated palm
    "./right.xml", # right rotated palm
    "./fist.xml"
]

cascade_list = []
for p in HAND_DETECTION_MODEL_PATH:
    cascade = cv2.CascadeClassifier(p)
    if cascade.empty(): raise RuntimeError("Failed to load palm cascade")
    cascade_list.append(cascade)

# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

cap = cv2.VideoCapture(0)

while(True):

    ret, current_buffer = cap.read()
    if max(current_buffer.shape[0], current_buffer.shape[1]) > 640:
        current_buffer = cv2.resize(current_buffer, (0,0), fx=0.5, fy=0.5)

    current_buffer_gray = cv2.cvtColor(current_buffer, cv2.COLOR_BGR2GRAY)
    current_buffer_gray = clahe.apply(current_buffer_gray)

    # detect hand in the masked gray image
    # the last parameter of detectMultiScale() is related to thereshold of detection

    detected_hand = []

    for c in cascade_list:
        detected_hand.extend(list(c.detectMultiScale(current_buffer_gray, 1.1, 3)))

    detected_hand = sorted(detected_hand, key = lambda t: t[2])

    if len(detected_hand) > 0:
        # continue
        detected_hand = detected_hand[-1]

        current_buffer = cv2.rectangle(current_buffer, (detected_hand[0], detected_hand[1]), \
            (detected_hand[0] + detected_hand[2], detected_hand[1] + detected_hand[3]), color, thickness) 

    cv2.imshow('...', current_buffer)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    pass

cap.release()
cv2.destroyAllWindows()