import cv2
import numpy as np
import glob

# Chessboard pattern is 9×6 INNER corners
CHECKERBOARD = (9, 6)

square_size = 0.025

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Scale by real square size
objp *= square_size

# Storage for points
objpoints = []  
imgpoints = [] 

images = glob.glob("Proj2_Images/*.png")

print(f"Found {len(images)} images")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"✔ Corners detected in {fname}")
        objpoints.append(objp)

        # Refine corner locations for higher accuracy
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        imgpoints.append(corners_refined)

        #Previdew detected corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        
cv2.destroyAllWindows()

# Camera calibration
print("\n🔧 Running camera calibration...")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== CAMERA CALIBRATION RESULTS ===")
print("Calibration RMS error:", ret)
print("\nCamera Matrix (K):\n", K)
print("\nDistortion Coefficients:\n", dist)
print("\nNumber of views used:", len(objpoints))

# Save calibration results
np.save("camera_matrix.npy", K)
np.save("dist_coeffs.npy", dist)

# Undistort a sample image
sample_img = cv2.imread(images[0])
h, w = sample_img.shape[:2]

# Get optimized camera matrix
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)

undistorted = cv2.undistort(sample_img, K, dist, None, new_camera_matrix)

cv2.destroyAllWindows()
