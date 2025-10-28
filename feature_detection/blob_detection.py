import cv2
import numpy as np

#Load image and covert it to grayscale
image = cv2.imread("images/example-image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initialize SIFT detector
sift = cv2.SIFT_create()

#Find keypoints
keypoints = sift.detect(gray, None)

#Re-initialize SIFT with tuned parameters
sift = cv2.SIFT_create(
    contrastThreshold=0.10,   
    edgeThreshold=20,        
    sigma=2.6,               
    nOctaveLayers=8          
)

#Tunable parameters
print("SIFT Parameters:")
print(f"contrastThreshold: {sift.getContrastThreshold()}")
print(f"edgeThreshold: {sift.getEdgeThreshold()}")
print(f"sigma: {sift.getSigma()}")
print(f"nOctaveLayers: {sift.getNOctaveLayers()}")

#Display information about the first few keypoints
for i, kp in enumerate(keypoints[:5]):
    print(f"Keypoint {i+1}:")
    print(f" - Coordinates: {kp.pt}")
    print(f" - Scale (size): {kp.size}")
    print(f" - Orientation (angle): {kp.angle}")
    print(f" - Response: {kp.response}")
    print(f" - Octave: {kp.octave}")

#Draw keypoints with their scales
output_image = cv2.drawKeypoints(
    image, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("SIFT Keypoints", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
