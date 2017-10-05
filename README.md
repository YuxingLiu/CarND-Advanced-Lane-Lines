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
[image4]: ./output_images/col_thresh1.png "Color Thresholding 1"
[image5]: ./output_images/col_thresh2.png "Color Thresholding 2"
[image6]: ./output_images/grad_thresh1.png "Gradient Thresholding 1"
[image7]: ./output_images/grad_thresh2.png "Gradient Thresholding 2"
[image8]: ./output_images/com_thresh1.png "Color/Gradient Thresholding 1"
[image9]: ./output_images/com_thresh2.png "Color/Gradient Thresholding 2"
[image10]: ./output_images/search1.png "Color/Search 1"
[image11]: ./output_images/search2.png "Color/Search 2"
[image12]: ./output_images/screen1.png "Color/Screen 1"
[image13]: ./output_images/screen2.png "Color/Screen 2"

---

## Camera Calibration

The code for this step is in the notebook [`Camera Calibration.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Camera%20Calibration.ipynb).  

A 9x6 chessboard is used for camera calibration. First, I prepared "object points" by assigning the (x, y, z) coordinates of the chessboard corners in the world, under the assumption that the chessboard is fixed on the (x, y) plane at z=0, and the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image are successfully detected. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then, I used the `cv2.calibrateCamera()` function to compute the camera calibration and distortion coefficients. I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

## Pipeline (single image)

### Distortion Correction

The code for this step is contained in the code cells [3]-[4] of the notebook [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb). 

Distortion correction was applied to one of the raw images as illustrated below: 

![alt text][image2]

### Perspective Transform

The code for this step is contained in the code cells [5]-[7] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).

Perspective transform was applied to rectify a undistorted image to a birds' eye view. The source and destination points are manually tuned as follows:

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

Motivated by [Peter Moran's Blog](http://petermoran.org/robust-lane-tracking/) on robust lane tracking, I explored three different color spaces, including HLS, HSV, and [CIELAB](https://en.wikipedia.org/wiki/Lab_color_space). Color thresholding was conducted on a **warped** image, which consists of the following steps:
1. Convert an RGB image to a single channel image on a new color space using `cv2.cvtColor()`.
2. Normalize the single channel image using [CLAHE](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) (Contrast Limited Adaptive Histogram Equalization).
3. Obtain a binary image by selecting the pixels within the range of 'threshold'.
4. Repeat the process for multiple color spaces and channels, and take the union of all binary images to get a multi-channel-thresholded binary image.

After tuning, I decided to implement three color channels with proper threshold values as follows:

| Color Space        | Channel   |  Threshold |
|:------------------:|:---------:|:----------:| 
| HLS                | L         | (240, 255) |
| HSV                | V         | (240, 255) |
| LAB                | B         | (170, 255) |

The multi-channel color thresholding was then verified on two test images:

![alt text][image4]
![alt text][image5]

#### Gradient Thresholding

In addition, gradient thresholding was applied on an **undistorted ** image, which consists of the following steps:
1. Convert an RGB image to grayscale image using `cv2.cvtColor()`.
2. Calculate the derivative in x and y directions using `cv2.Sobel`.
3. Based on the thresholds of x and y gradients, gradient magnitude and direction, obtain 4 binary images `sx_binary `, `sy_binary`, `mag_binary`, and `dir_binary`.
4. Combine those individual thresholds and get a gradient thresholded binary image, according to:

```python
grad_binary[((sx_binary == 1) & (sy_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

After tuning, `sobel_kernel=3` and the following threshold values are used:

| Gradient  |  Threshold |
|:---------:|:----------:| 
| x         | (30, 100)  |
| y         | (30, 100)  |
| magnitude | (50, 100)  |
| direction | (0.7, 1.3) |

The gradient thresholding was then verified on two test images:

![alt text][image6]
![alt text][image7]

#### Color and Gradient Thresholding

Finally, the combination of color and gradient thresholding was adopted, resulting in a union of two binary images as shown below:

![alt text][image8]
![alt text][image9]

### Lane Lines Detection

The code for this step is contained in the code cells [14]-[16] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).

First, I use a histogram and sliding window as a starting point to identify the pixels of lane lines. Once the left and right line pixels are found, fit a second order polynomial to each using `np.polyfit`.

If the lane lines were successfully found in the previous frame, we could use a look-ahead filter to search within a window around the previous detection. The two lane finding methods are illustrated on two test images:

![alt text][image10]
![alt text][image11]

### Curvature and Offset

The code for this step is contained in the code cell [17] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

![alt text][image12]
![alt text][image13]



## Pipeline (project video)

### Line Class

The code for this step is contained in the code cell [18] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).

### Process Pipeline

The code for this step is contained in the code cells [19]-[20] of [`Test Video Pipeline.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline.ipynb).


Here's a link to [project video result](./test_videos_output/project_video.mp4).


## Pipeline (challenge video)

The code for this section is in the notebook [`Test Video Pipeline_challenge.ipynb`](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines/blob/master/Test%20Video%20Pipeline_challenge.ipynb).

Here's a link to [challenge video result](./test_videos_output/project_video.mp4).

Here's a link to [project video result](./test_videos_output/project_video2.mp4) using the new pipeline.
