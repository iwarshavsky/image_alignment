# image_alignment

This was an exercise I did in Image Processing. Its goal was to align and blend two images together, the first being in low resolution, and the second being a high-resolution part of the first image which has undergone some projective transformation. The main techniques I used to solve this included feature detection with the Harris corner detector, feature extraction using a variant of MOPS descriptors, matching of said features, calculation of a transformation using 4 pairs of points, the RANSAC algorithm for detection of a suitable transformation for the alignment and finally backward warping and blending. 

This was an exercise in Image Processing that I (mistakenly) implemented all by myself, without the use of external libraries like OpenCV (apart from NumPy and the occasional convolution with a Sobel kernel using SciPy).

The report explaining the process is attached as a PDF file.

These were the original images:

Low resolution                                                                                                                | High Resolution
----------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------------------------------------------------------------------------------------------------------------------------------

![lake low res](https://github.com/iwarshavsky/image_alignment/blob/main/img/lake_low_res.jpg?raw=true "Low resolution lake") | ![lake high res](https://github.com/iwarshavsky/image_alignment/blob/main/img/lake_high_res.png?raw=true "High resolution lake with projective transformation")
<p float="left">
  <img src="./img/lake_low_res.jpg?raw=true" width="49%" />
  <img src="./img/lake_high_res.png?raw=true" width="49%" /> 
</p>
