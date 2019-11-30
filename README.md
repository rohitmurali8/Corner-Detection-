# Corner-Detection-
Detection of corners in an image without using opencv functions from scratch

PSEUDOCODE:

1-	Find approximation of the gradients of the image in the X and Y directions by convolving with the sobel kernels.

2-	Obtain the approximate second derivative by multiplying these gradients with each other.

3-	Convolve a gaussian filter with these second derivatives.

4-	Obtain the corner strength which is obtained by the formula 
  R = det(M) – k*(trace(M))**2
  Here, M is the hessian matrix which is given by convolution of a gaussian filter with second derivative of the image in the X and Y directions along the diagonal and convolution of the gaussian filter with product of X and Y derivative of the image as the rest of the elements.
 
5-	Pass the obtained corner strength matrix into a function which calls the non-maxima suppressor function for these points in the matrix.

6-	Compare the matrix returned by the non-maxima suppressor function with the threshold value and draw circles around the points in the image where the corner strength is greater than the threshold. 
