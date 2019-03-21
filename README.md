## Advanced-Lane-Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration2.jpg "Undistorted"
[image2]: ./test_images/test3.jpg "Road Transformed"
[image3]: ./output_images/binary_combo.jpg "Binary Example"
[image4]: ./output_images/warped_binary.jpg "Warp Example"
[image5]: ./output_images/two_order.jpg "Fit Visual"
[image6]: ./output_images/output_image.jpg "Output"
[image7]: ./output_images/output_all_image.jpg "All Output"
[video1]: ./output_videos/project_video.mp4 "Video"
[video2]: ./output_videos/challenge.mp4 "Challenge"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "`./examples/01-CameraCalibration.ipynb`".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and sobel thresholds to generate a binary image in `./examples/02-ColorTransform.ipynb`.  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

In the above color_binary, green is the result of sobel operator, and blue is the one of color thresholds.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `bird_eye()`, which appears in No.11 cell in the file `./examples/03-PerspectiveTransform.ipynb`.  The `bird_eye()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
imshape = img.shape
X, Y = imshape[1], imshape[0]
offset = 200
dst = np.float32([
    (offset, Y),
    (offset, 0),
    (X-offset, 0),
    (X-offset, Y)
])
src = np.float32(
    [[(200, Y)], 
    [(X/2-40, Y*.6)], 
    [(X/2+40, Y*.6)], 
    [(1200, Y)]])
dst = np.float32(
    [[(offset, Y)],
    [(offset, 0)],
    [(X-offset, 0)],
    [(X-offset, Y)]])
```

[[[ 200.  720.]
  [ 600.  432.]
  [ 680.  432.]
  [1200.  720.]]]
[[ 200.  720.]
 [ 200.    0.]
 [1080.    0.]
 [1080.  720.]]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 200, 720        | 
| 600, 432      | 200, 0      |
| 680, 432     | 1080, 0      |
| 695, 720      | 1080, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of the lane in lines 189 through 193 in my code in `Advanced-Lane-Lines.py`.
In lines 218 through 222, I calculated the positon of the vehicle with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 195 through 212 in my code in `Advanced-Lane-Lines.py` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

Here is output of all test images:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [project_video.mp4 made by using my pipeline](./output_videos/project_video.mp4)

I also tried finding lane of challenge_video.mp4.
But the result is not good.

Here's a [challenge_video.mp4 made by using my pipeline](./output_videos/challenge_video.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
