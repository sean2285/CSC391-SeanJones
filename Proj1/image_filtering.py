import cv2
import numpy as np

def apply_convolution(image, kernel):
    """Perform convolution without using OpenCV filter2D."""
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    # Pad image to handle borders
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image)

    # Iterate through image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y+k_height, x:x+k_width]
            output[y, x] = np.clip(np.sum(region * kernel), 0, 255)
    return output.astype(np.uint8)

# Define kernels
box_filter = (1/9) * np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]])

gaussian_filter = (1/16) * np.array([[1,2,1],
                                     [2,4,2],
                                     [1,2,1]])

sobel_horizontal = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]])

sharpen_filter = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])

# Choose which filter to test
current_filter = gaussian_filter

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply convolution
    filtered = apply_convolution(gray, current_filter)

    # Show original and filtered images
    cv2.imshow("Original (grayscale)", gray)
    cv2.imshow("Filtered", filtered)

    # Press keys to switch filters
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        current_filter = box_filter
        print("Switched to Box filter")
    elif key == ord('2'):
        current_filter = gaussian_filter
        print("Switched to Gaussian filter")
    elif key == ord('3'):
        current_filter = sobel_horizontal
        print("Switched to Sobel horizontal filter")
    elif key == ord('4'):
        current_filter = sharpen_filter
        print("Switched to Sharpen filter")
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()