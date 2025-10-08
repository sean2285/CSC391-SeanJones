1. contrast_stretch(img, r_min, r_max)
- Enhances the contrast of an image by linearly mapping pixel intensities from the range `[r_min, r_max]` to the full `[0, 255]` range.
- Brightens dark regions and darkens bright regions to better utilize the full intensity spectrum.

2. calculate_histogram(img, bins)
- Computes the histogram of a grayscale image and its normalized probability distribution.
- Returns 
  - counts: Raw frequency of pixels in each bin  
  - dist: Normalized distribution

3. equalize_histogram(img)
- Redistributes image intensities using histogram equalization to improve global contrast.
- Expands intensity variations, making hidden details more visible.

4. median_filter(img, size=3)
- Applies a non-linear median filter to reduce salt and pepper noise while preserving edges.
- Each pixel is replaced by the median of its surrounding neighborhood

5. calculate_gradient(img)
- Computes the gradient magnitude and direction of an image using Sobel operators.
- Returns
  - grad_magnitude: Strength of intensity changes
  - grad_angle: Edge orientation in degrees

6. sobel_edge_detector(img, threshold)
- Performs basic edge detection using Sobel filters and a threshold on gradient magnitude.
- Highlights all strong edges in the image; results in thicker, binary edge maps.

7. directional_edge_detector(img, direction_range)
- Detects edges that fall within a specified gradient direction range
- Useful for isolating edges with a particular orientation.




