import cv2
import numpy as np
import glob

# Chessboard dimensions
CHECKERBOARD = (9, 6)       
square_size = 0.025         

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  
imgpoints = []  

# Load images
images = glob.glob("Proj2_Images/*.png")
print(f"\nFound {len(images)} images.")

if len(images) == 0:
    print("No images found")
    exit()

# Detect corners in images
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"✔ Corners detected: {fname}")
        objpoints.append(objp)

        # Refining corner accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_refined)
    else:
        print(f"✖ Corners NOT found: {fname}")

if len(objpoints) < 5:
    print("\nNot enough valid images")
    exit()

# Camera calibration

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== CAMERA CALIBRATION RESULTS ===")
print("Calibration RMS error:", ret)
print("\nCamera Matrix (K):\n", K)
print("\nDistortion Coefficients:\n", dist)
print("\nNumber of valid images used:", len(objpoints))

# Save calibration results
np.save("camera_matrix.npy", K)
np.save("dist_coeffs.npy", dist)


# Undistort a sample image and save
sample_img = cv2.imread(images[0])
h, w = sample_img.shape[:2]
new_cam, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
undistorted = cv2.undistort(sample_img, K, dist, None, new_cam)
comparison = np.hstack((sample_img, undistorted))

cv2.imwrite("comparison_distorted_vs_undistorted.png", comparison)

cv2.imshow("Distorted (Left) vs Undistorted (Right)", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
