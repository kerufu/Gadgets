# This program is used for calculating intrinsic and extrinsic parameters of camera.
# The camera intrinsic matrix and distortion coefficient are written into a yaml file.
# The number of inner corners per a item row and column may need to be changed. Default is 9 * 6.
# The image path and image style may need to be changed.

import numpy as np
import cv2
import glob
import yaml

# Default number of inner corners per a item row and column
row = 9
col = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((col*row,3), np.float32)
objp[:,:2] = np.mgrid[0:row,0:col].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# get image names
images = glob.glob('calibration_image/*.png')  # The image path and image style may need to be changed

# set window size
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("img", 800, 800)

imageSize = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageSize = gray.shape[::-1]
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (row,col), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (row,col), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None, flags=16384) # flags=16384
print(dist.size)

# to show undistorted image
img = cv2.imread('calibration_image/0.png')  # The path may need to be changed
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('img', dst)
cv2.waitKey(5000)
cv2.imwrite('calibresult.png', dst)

# organize intrinsic matrix
intrinsic = []
for i in range(0, 3):
    intri_row = []
    for j in range(0, 3):
        intri_row.append(float(mtx[i][j]))
    intrinsic.append(intri_row)

# organize distortion coefficient
dist_coeff = []
for i in range(0, dist.size):
    dist_coeff.append(float(dist[0][i]))

data = {"intrinsic_matrix": intrinsic, "dist_coeff": dist_coeff}

# write data into .yaml file
fname = "data.yaml"
with open(fname, "w") as f:
    yaml.dump(data, f)
    
cv2.destroyAllWindows()