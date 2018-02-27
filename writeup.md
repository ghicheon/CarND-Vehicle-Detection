**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]:   ./writeup_images/image59_notcar.png
[image1_1]: ./writeup_images/image0110_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
It can be found in lines 153 through 171 of the files called 'final.py'    
I extracted HOG features by calling skimage.feature.hog function. 
I used 9 for orient ,8 for pixel per cell ,2 cell per block and "ALL" hog channels. I tried several color space and found out YCrCb gave me the best result.    



non-car example    
![alt text][image1]

car example
![alt text][image1_1]


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
I tried some combinations of hog parameters. Definitely,using 3 channels gave me much improvement. I got better result under the sun by setting transform_sqrt argument to True.     
Actually,at first It does NOT work AT ALL.. It was because of reading png files. I had a hard time to use both mpimg() and imread() and gave up using them together. I only used cv2.imread() in order to read image files! After I apply this to every code including predicting, it started to be working magically!    
I got over 99.5% test accuracy.(well.. I also used spatial features and histogram features. But,I'm sure that this accuracy mainly came from hog features!)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
I used vehicles.zip and non-vehicles.zip files for training SVM. The number of vehicles and non vehicles are 8792 and 8968 respectively.    
First, I extracted features from these image files. The function extract_features() did all for it. I used hog features as well as color histogram features and spatial features.   
Second, I divided extracted features data into 2 parts. 80% of data is for traing. 20% of data is for testing.    
Third,I nomalized the data using StandardScaler() in order to make all features influence to the result evenly.   
I took advantage of pickle of python to save time.If I set PICKEL_READY to True,training is skipped.    
The code can be found in car_detect_init() function of the file called final.py

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I got around 0.875 overlap windows by setting cells_per_step to 1. It gave me huge better result in video.But it took almost twice time when processing video file.      
I spent some time to finding good value for scale. I found out default value 1.5 was not bad in my experience.  There might be better value. But,I didn't think it's worth finding it. I think it's better to put effort into making good tracking algorithm. Because of this, I decided to use 1.5 for scale.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
Original sliding window took lots of time for training.I didn't get hog features for every window.Instead, I got hog feature 1 time per frame and used small part(window size) of it. In fact, My code is based on find_car() of the lecture. I made it work in open cv by deleting dividing by 255(Wow... I spent too much time for it).    
Upper part of car_detect() is for it.

  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Whenever my classifier says positive detection, I added the rectangle to box_list list. I used heatmap,threshhold,and label to find car position.
I spent so much time on getting a good result only with these tools.But It was not possible!! Finally, I realized that I have to make a code for tracking cars by using specific data structure!!   

I created Car class. It represents one car. 'ref' member is for maintaining the reference count.If a car is detected several times, it's worth showing it.    
'ref' is also used to show persistant box. When ref is more than 10,this one must be a car definitely.I added 10 more reference in that case. It made it possible to show a car even if it's not detected for a while.

Before I draw a rectangle,I check ref memger of Car object. If it's less than 5, I ignore it at the moment becuase it might be a false positive.
I ignored the detected object when the width is less than 30 pixel. it's too narrow to be a car.    
I also considered the ratio of width and height. For example, if the width is 100 and the height is 160, it's not a car. 

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I only considered previous and current frame. If many frames for a car are used and getting average of them, the rectangle will be robust.   

 I'm thinking of (after some frames..) adding a new car only from specitic area such as bottom line and upper line. A car can't appear in the middle of road without crossing upper and bottom line. It can reduce false positive.




Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

