# image_alignment

The goal of this exercise was to align and blend two images together, the first being in low resolution, and the second being a high-resolution part of the first image which has undergone some projective transformation. The main techniques I used to solve this included feature detection with the Harris corner detector, feature extraction using a variant of MOPS descriptors, matching of said features, calculation of a transformation using 4 pairs of points, the RANSAC algorithm for detection of a suitable transformation for the alignment and finally backward warping and blending. 

This was an exercise in Image Processing that I (mistakenly) implemented all by myself, without the use of external libraries like OpenCV (apart from NumPy and the occasional convolution with a Sobel kernel using SciPy).

The report explaining the process is attached as a PDF file.
