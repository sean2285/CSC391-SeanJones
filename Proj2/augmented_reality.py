import cv2
import numpy as np

# Load camera calibration data
K = np.load("camera_matrix.npy")
dist = np.load("dist_coeffs.npy")

# Chessboard dimensions
CHECKERBOARD = (9, 6)
square_size = 0.025 

# Load test image
img = cv2.imread("test_image.png") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prepare 3D object points for the chessboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Detect corners
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if not ret:
    print("Chessboard corners not found")
    exit()

# Refine corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# Estimate pose
ret, rvec, tvec = cv2.solvePnP(objp, corners_sub, K, dist)

print("\nRotation Vector (rvec):\n", rvec)
print("\nTranslation Vector (tvec):\n", tvec)

# Define cube corners in 3D space
cube_size = square_size * 3 
cube_points = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, -cube_size],
    [cube_size, 0, -cube_size],
    [cube_size, cube_size, -cube_size],
    [0, cube_size, -cube_size]
], dtype=np.float32)

# Project cube corners into the image
imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, K, dist)

# Convert to integer pixel coords
imgpts = np.int32(imgpts).reshape(-1, 2)

# Function to draw the cube
def draw_cube(image, pts):
    # bottom square
    img = cv2.drawContours(image, [pts[:4]], -1, (0, 255, 0), 3)

    # top square
    for i in range(4):
        img = cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (255, 0, 0), 3)

    img = cv2.drawContours(img, [pts[4:]], -1, (0, 0, 255), 3)

    return img

output = draw_cube(img, imgpts)

cv2.imwrite("AR_result.png", output)
print("\nSaved AR_result.png")
