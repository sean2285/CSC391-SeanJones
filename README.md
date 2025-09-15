# CSC391-SeanJones
This project demonstrates three main concepts: signal processing, image transformation, and optics.

First, a sine wave signal is generated, sampled, and quantized. Gaussian noise can be added to the samples, and error metrics such as mean squared error (MSE), root mean squared error (RMSE), and peak signal-to-noise ratio (PSNR) are calculated. The results are visualized by plotting the continuous signal, noisy sampled points, and the quantized signal. A second example shows signal sampling and quantization without noise, along with the quantization error.

Second, an image is loaded and transformed using a perspective warp. The original image is shown alongside a “leaning card” version of the same image created by mapping its corners to new locations. This shows how perspective transformations can rotate, translate, and skew images.

Third, the thin lens law from optics is implemented to calculate the image distance for different object distances and focal lengths. Log-log plots of image distance versus object distance are generated for several lenses. The aperture formula is also applied to compute aperture diameters for real-world camera lenses.