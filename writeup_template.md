---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Img1.png "Undistorted"
[image2]: ./output_images/Img2.png "Undistorted"
[image3]: ./output_images/Img3.png "Undistorted"
[image4]: ./output_images/Img4.png "Undistorted"
[video1]: ./output_images/uno.mp4 "Undistorted"
[image5]: ./output_images/Img5.png "Undistorted"
[image6]: ./output_images/Img6.png "Undistorted"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Img1.png "Undistorted"
[image2]: ./output_images/Img2.png "Undistorted"
[image3]: ./output_images/Img3.png "Undistorted"
[image4]: ./output_images/Img4.png "Undistorted"
[video1]: ./output_images/uno.mp4 "Undistorted"
[image5]: ./output_images/Img5.png "Undistorted"
[image6]: ./output_images/Img6.png "Undistorted"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images in the dataset, but after many iterations I tested an idea expresed in the forums, that "Considering also that dataset consists from a sequence of similar images from video stream, I've used just every 10th image from the dataset" that was corrected in another post to every 5th image (https://discussions.udacity.com/users/driveWell) I keep testing this idea, with better results that using the whole database. Nevertheless, I have used the full `non-vehicle` images despite the fact that there is an unbalanced training set, to reduce the rate of false positives, also, after a lot of testing iterations.

I also explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but the key parameters in this project turned to be only the color space (both YCrCb and YUV are the best ones) and window size (128x128). I was unable to improve the results appreciably, varying the other parameters, so I left most of them fixed and worked with the two mentioned.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using Spatial, Color Histogram and Hog features. The results are saved in a file to be used later by the pipeline.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is based in the Hog Sub-sampling Window Search algorithm and is the function `find_cars()`in the code. 

I decided to try several scales and eliminate those that were responsible of false positives, the scales finally used are 1, 1.3, 1.5, 1.7

After several experiments using different values of overlap windows, I set the parameter to the recommended minimun 0.65.

![alt text][image2]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a decent result.  Here are some example images:

![alt text][image3]

![alt text][image5]

![alt text][image6]


As I said before, the most important decision as to the classifier has been which of the images in the dataset to use.
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result]![alt text][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

To eliminate false positives, the class `Heatmap_nFrames` using `collections.deque(maxlen=15)` compute the heatmap, using the last 15 frames. Another improvement not tested is to give greater weight to new frames compared to older ones, but it was not necessary.

As a consequence of using the last 15 frames, the threshold should also be fixed at a relatively high value: 40

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are a frame and their corresponding heatmap and the output of `scipy.ndimage.measurements.label()`  :

![alt text][image3]


### Here the resulting bounding boxes are drawn onto the same frame in the serie:
![alt text][image4]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Following the instructions, code and recommendations of the lesson, the truth is that I got to a pipeline that worked, but the process of improvement and testing of parameters has been very laborious and complicated.

I think the dataset I used was not of high quality and perhaps should have spent much more effort in using the new dataset of images or performing data_augmentation.

Although the final video, I think it has enough quality to meet the requirements of the project, is not robust enough for other environmental circumstances.

