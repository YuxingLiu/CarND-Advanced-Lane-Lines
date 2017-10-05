# Advanced Lane Finding Project

This repository presents an image processing pipeline for detecting lane lines in a variety of conditions, including changing road surfaces, curved roads, and variable lighting.

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation. 

[//]: # (Image References)

[image1]: ./output_images/camera_cal.png "Camera Calibration"
[image2]: ./output_images/dist_corr.png "Distortion Correction"
[image3]: ./output_images/perspect_trans.png "Perspective Transform"

---

## Camera Calibration

The code for this step is contained in the notebook [`Camera Calibration.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Camera%20Calibration.ipynb).  

A 9x6 chessboard is used for camera calibration. First, I prepared "object points" by assigning the (x, y, z) coordinates of the chessboard corners in the world, under the assumption that the chessboard is fixed on the (x, y) plane at z=0, and the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image are successfully detected. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then, I used the `cv2.calibrateCamera()` function to compute the camera calibration and distortion coefficients. I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

## Pipeline (single image)

### Distortion Correction

The code for this step is contained in the code cells [3]-[4] of the notebook [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb). 

Distortion correction was applied to one of the raw images as illstrated below: 

![alt text][image2]

### Perspective Transform

The code for this step is contained in the code cells [5]-[7] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).

Perspective transform was applied to rectify a undistorted image to a birds' eye view. The source and destination points are manully tuned as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 584, 458      | 256, 0        | 
| 209, 720      | 256, 720      |
| 1113, 720     | 1024, 720     |
| 698, 458      | 1024, 0       |

I use the `cv2.getPerspectiveTransform()` function to compute the transform matrix and `cv2.warpPerspective()` to warp the image. The perspective transform was verified by drawing the `src` and `dst` points onto a straight-lines test image and its warped counterpart as follows:

![alt text][image3]

### Binary Image

The code for this step is contained in the code cells [8]-[13] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).

I use both color and gradient thresholds to create a binary image containing likely lane pixels.

#### Color Thresholding

Motivated by [Peter Moran's Blog on Robust Lane Tracking](http://petermoran.org/robust-lane-tracking/), I explored three different color spaces, including HLS, HSV, and [CIELAB](https://en.wikipedia.org/wiki/Lab_color_space). Color thresholding was conducted on an warped image, which consists of the following steps:
1. Convert an RGB image to a single channel image on a new color space using `cv2.cvtColor()`.
2. Normalize the single channel image using [CLAHE](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) (Contrast Limited Adaptive Histogram Equalization).
3. Obtain a binary image by selecting the pixels within the range of 'threshold'.
4. Repeat the process for multiple color spaces and channels, and take the union of all binary images to get an multi-channel-thresholded binary image.

After tuning, I decided to use the following three channels and associated threshold values:

| Color Space        | Channel   |  Threshold |
|:------------------:|:---------:|:----------:| 
| HLS                | L         | (240, 255) |
| HSV                | V         | (240, 255) |
| LAB                | B         | (170, 255) |


#### Gradient Thresholding


#### Color and Gradient Thresholding


### Lane Lines Detection


#### Sliding Window


#### Look-Ahead Filter


### Curvature and Offset



## Pipeline (project video)



### `Line()` Class


## Pipeline (challenge video)
