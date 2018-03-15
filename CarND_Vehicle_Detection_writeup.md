## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/CarAndNotCar.jpg
[image2]: ./output_images/frame1.png
[image3]: ./output_images/frame2.png
[image4]: ./output_images/frame3.png
[image5]: ./output_images/frame4.png
[image6]: ./output_images/frame5.png
[image7]: ./output_images/frame6.png
[image8]: ./output_images/Serie_frame1.png
[image9]: ./output_images/Serie_frame2.png
[image10]: ./output_images/Serie_frame3.png
[image11]: ./output_images/Serie_frame4.png
[image12]: ./output_images/Serie_frame5.png
[image13]: ./output_images/Serie_frame6.png
[image14]: ./output_images/SlidingWindow.png
[image15]: ./output_images/Hog_vis.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG feature of a single image is extracted by calling the ["get_hog_features"](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/67ec1117045f814ef2d42f0b2b4f6f1f01e808d7/lesson_functions.py#L20) function. Then I call function ["extract_features"](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/67ec1117045f814ef2d42f0b2b4f6f1f01e808d7/lesson_functions.py#L114) to get HOG feature of all images in the data set.

I started by reading in all the `vehicle` and `non-vehicle` images [here](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/67ec1117045f814ef2d42f0b2b4f6f1f01e808d7/Training_Model.py#L63).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:




#### 2. Explain how you settled on your final choice of HOG parameters.

To choose HOG parameters, I tried various combinations of parameters and choose the 'LUV' color space because this color space seperates the luminance component from the other two color components. The other three parameters are chosen such that the HOG feature is representative and concise as well. I grabbed two images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.

![alt text][image15]

The other HOG parameters are chosen as follows:

<pre><code>
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
</code></pre>

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


To train a linear SVM, first I attract HOG-feature combined with historgram and spatial feature of all images in training set using function [extract_features](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/becb31638bcbb64c881d689348e4f90ceccd00f4/Training_Model.py#L71). Then the SVM is fit to the extracted features to determine the classifier. The classifier has an accuracy of 98.96 %

In the next step I test the classifier with 10 labels
<pre><code>
Test Accuracy of SVC =  0.9896
My SVC predicts:  [0. 1. 1. 1. 0. 1. 1. 1. 0. 0.]
For these 10 labels:  [0. 1. 1. 1. 0. 1. 1. 1. 0. 0.]
0.01602 Seconds to predict 10 labels with SVC
</code></pre>

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use the function [find_car](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/9f4264e3ea43335af99936a2cc4c163e36da1bc8/lesson_functions.py#L165), provived in the lessons for the implementation of sliding window with ystart = 300 and ystop = 700. That means, we are finding cars only in the lower half of the frame. The scales for search varies from 1.25 to 3.5 to find cars with different sizes and at different distances. The picture below shows an implementation of sliding windows with different sizes and overlap:

![alt text][image14]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on seven scales ([1.25, 1.75, 2.0, 2.3, 2,7, 3.0,3.5]) using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
---
 
As we see in the pictures, a lot of false positives were detected. To reject these detections, we need a threshold to filter out the unexpected detection, as well as more scales to detect objects with different size.
 
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=l11FoZUxuu8&feature=youtu.be) or [here](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/master/output_test_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The queue is implemented in the class [Window_buf](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/724cf433e2fc4a3923a70daa8b86d1a53cb8bf67/Finding_Car.py#L16)

I use a queue to record the position of positive detections in last 5 subsequent frames. After a frame is processed and some positive positions are detected. These detected positions are pushed into the queue. When the queue is full, the oldest detected positions are pop out of the queue. Then I create a heatmap from the detected positions, registered in the queue. This queue works as a buffer and smooths the detection of car through out the video. In the next step I threshold the heatmap to filter out the false detections, as well as to identify vehicle position. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The video pipeline is implemented in the functions [Find_Car_In_Frame](https://github.com/truongconghiep/CarND-Vehicle-Detection/blob/5de3d8ab7eb323841ce15c1d471fa7e9728d9e1d/Detecting_In_Video.py#L15)


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps and outputs with labels and detection smoothing:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

