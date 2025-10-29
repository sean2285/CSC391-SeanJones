import cv2
import numpy as np
import os 

#Load image and covert it to grayscale
image = cv2.imread("images/example-image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply transformations to create a second image
rows, cols = gray.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1.2)
transformed = cv2.warpAffine(image, M, (cols, rows))

tx, ty = 30, 20
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
transformed = cv2.warpAffine(transformed, translation_matrix, (cols, rows))

#Save new image
os.makedirs("images", exist_ok=True)
transformed_path = "images/example-image-transformed.jpg"
cv2.imwrite(transformed_path, transformed)

image2 = cv2.imread(transformed_path)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#Initizalize SIFT with custom parameters (values altered for best appearing results)
sift = cv2.SIFT_create(
    contrastThreshold=0.075,   
    edgeThreshold=2.5,        
    sigma=3,               
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
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

#Print information
print(f"\nOriginal Image Total keypoints: {len(keypoints)}")
print(f"Original Image Descriptor shape: {descriptors.shape}")
print(f"Transformed Image Total keypoints: {len(keypoints2)}")
print(f"Transformed Image Descriptor shape): {descriptors2.shape}")

#Utilize Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

#Draw top 50 matches
matched_img = cv2.drawMatches(
    image, keypoints,
    image2, keypoints2,
    matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

#Display results
cv2.imshow("Top 50 SIFT Feature Matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#One randomly selected keypoint
selected_kp = keypoints[22]
descriptor = descriptors[22]
x, y = np.int32(selected_kp.pt)
size = int(selected_kp.size)

#Plot all keypoints
output_image = cv2.drawKeypoints(
    image, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

#Highlight one keypoint for observation (blue rectangle surrounding the point)
cv2.circle(output_image, (x, y), size, (0, 255, 0), 2)
cv2.rectangle(output_image, (x - size, y - size), (x + size, y + size), (255, 0, 0), 1)

cv2.imshow("SIFT Keypoints + Selected Region", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Print the descriptor of the selected keypoint
print(descriptor)