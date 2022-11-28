import cv2
import numpy as np
import glob

#parameters to enable ORB use KNN
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary
kp_matcher=cv2.FlannBasedMatcher(index_params,search_params)

ratio_test_threshold=0.5
match_threshold=10


#process test image
orb_Detector = cv2.ORB_create(nfeatures=1000)
file_paths = glob.glob("test.jpeg")[0]
im = cv2.imread(file_paths)
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
test_kp,test_des=orb_Detector.detectAndCompute(im,None)

#start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


#keep detecting test image
while True:
    ret, image = cap.read()
    if not ret:
      break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp,des=orb_Detector.detectAndCompute(gray,None)
    all_matches=kp_matcher.knnMatch(des,test_des,k=2)
    image = cv2.drawKeypoints(image, kp, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    matches=[]
    #ratio test
    for m in all_matches:
        if len(m)==2:
            if m[0].distance < ratio_test_threshold*m[1].distance:
                matches.append(m[0])
    if len(matches)>match_threshold:
        try:
            qp = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            tp = np.float32([test_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            test_border=np.float32([[[0,0],[0,im.shape[0]-1],[im.shape[1]-1,im.shape[0]-1],[im.shape[1]-1,0]]])
            camera_border=cv2.perspectiveTransform(test_border,H)
            image = cv2.polylines(image, [np.int32(camera_border)], True, (0,0,255),3, cv2.LINE_AA)
        except Exception as e:
            print(e)
    else:
        print("Not enough matches")
    #print(len(matches1))
    cv2.imshow("emotion", image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()