import cv2
import numpy as np

#Load image and covert it to grayscale
image = cv2.imread("images/example-image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initizalize SIFT with custom parameters (values altered for best appearing results)
sift = cv2.SIFT_create(
    contrastThreshold=0.0225,   
    edgeThreshold=2.5,        
    sigma=2,               
    nOctaveLayers=4         
)

#Tunable parameters
print("SIFT Parameters:")
print(f"contrastThreshold: {sift.getContrastThreshold()}")
print(f"edgeThreshold: {sift.getEdgeThreshold()}")
print(f"sigma: {sift.getSigma()}")
print(f"nOctaveLayers: {sift.getNOctaveLayers()}")

#Find keypoints
keypoints, descriptors = sift.detectAndCompute(gray, None)
print(f"\nTotal keypoints: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")

#Display information about the first few keypoints
for i, kp in enumerate(keypoints[:5]):
    print(f"Keypoint {i+1}:")
    print(f" - Coordinates: {kp.pt}")
    print(f" - Scale (size): {kp.size}")
    print(f" - Orientation (angle): {kp.angle}")
    print(f" - Response: {kp.response}")
    print(f" - Octave: {kp.octave}")

#One randomly selected keypoint
selected_kp = keypoints[7]
descriptor = descriptors[7]
x, y = np.int32(selected_kp.pt)
size = int(selected_kp.size)

#Plot all keypoints
output_image = cv2.drawKeypoints(
    image, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

#Highlight one keypoint for observation
cv2.circle(output_image, (x, y), size, (0, 255, 0), 2)
cv2.rectangle(output_image, (x - size, y - size), (x + size, y + size), (255, 0, 0), 1)

cv2.imshow("SIFT Keypoints + Selected Region", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Print the descriptor of the selected keypoint
print(descriptor)

visualize_descriptor(descriptor)
