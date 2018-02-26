#######################################################################
# Udacity Self Driving Car Nanodegree 
#
# Project 5: Vehicle Detection and Tracking
# by Ghicheon Lee 
#
# date: 2018.2.26
#######################################################################

#I referenced a lot of codes and hints from "Vehicle Detection" Lecture!

import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import pickle
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pickle
from moviepy.editor import VideoFileClip
import os 
#import sys

# save time for training of SVC model, camera calibration, and scaler.
# pickle/dist_pickle.p  , pickle/svc.p ,  pickle/X_scaler.p
#PICKLE_READY = False
PICKLE_READY = True

#just for debug & writeup report.
debug=False
#debug=True

##################################################################
color_space = 'YCrCb'# Can be RGB, HSV, LUV, HLS!, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" #0 # Can be 0, 1, 2, or "ALL"

spatial_size = (32, 32) # Spatial binning dimensions
hist_size = 32    # Number of histogram bins

spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400,656]
##################################################################


# for pickle
svc = None            #SVC model
X_scaler = None       #Nomalization of features
dist=None
mtx=None


#how many continuous frames do we maintain for calculating "car detection"?
HEAT_LST_LIMIT = 10
valid_frames=7

#used in car_detect().
scale = 1.5

SMALL_DATASET = False
#SMALL_DATASET = True

is_video = False    #!!!



# maintain some continuous frames. 
heat_list = []



#list of founded cars.
#valid cars in gCars only have chances to be shown!!
cars=list()

#############################################
# class Car represents a found "car".
#############################################
class Car(object):
    # a: upper left point
    # b: bottom right point
    def __init__(self, a,b,ref=0):
        self.a = a
        self.b = b

        self.x1 = a[0]
        self.y1 = a[1]
        self.x2 = b[0]
        self.y2 = b[1]

        self.x = a[0]  #of upper left point
        self.y = a[1]  #of upper left point
        self.xcenter = (b[0] + a[0]) //2
        self.ycenter = (b[1] + a[1]) //2
        self.width = b[0] - a[0]
        self.height = b[1] - a[1]
        self.ref = ref
    def update(self,a,b):
        __init__(a,b)

    #get the distance between this object and argument object 
    #unit:pixel
    def distance(self,a,b):
        xcenter = (b[0] + a[0]) //2
        ycenter = (b[1] + a[1]) //2
        val = np.sqrt(  (xcenter - self.xcenter)**2 + (ycenter - self.ycenter)**2)
        print("distance:", val)
        return val

    def inc_ref(self):
        self.ref += 1
    def dec_ref(self):
        self.ref -= 1

    # the number of valid object must be having 5 refrences  at least.
    def valid(self):
        return self.ref >= 5   

############################################
# reference:  image size (720,1280)
############################################


def convert_color(img):
    global color_space
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: 
        feature_image = np.copy(img)      

    return feature_image 

def get_hog_features(img ,vis=False ,feature_vec=True):
    global orient, pix_per_cell, cell_per_block

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img):
    global spatial_size
    color1 = cv2.resize(img[:,:,0], spatial_size).ravel()
    color2 = cv2.resize(img[:,:,1], spatial_size).ravel()
    color3 = cv2.resize(img[:,:,2], spatial_size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img):
    global hist_size
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], hist_size)
    channel2_hist = np.histogram(img[:,:,1], hist_size)
    channel3_hist = np.histogram(img[:,:,2], hist_size)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector

    return hist_features

def extract_features(imgs): 
    global color_space
    global spatial_size, hist_size
    global orient,pix_per_cell, cell_per_block, hog_channel
    global spatial_feat, hist_feat, hog_feat

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        image = cv2.imread(file)

        img_features= single_img_features(image)

        features.append(img_features)

    # Return list of feature vectors
    return features
    

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to extract features from a single image window
def single_img_features(img):
    global color_space
    global spatial_size, hist_size
    global orient,pix_per_cell, cell_per_block, hog_channel
    global spatial_feat, hist_feat, hog_feat

    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img)
    #3) Compute spatial features if flag is set

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel],
                                            vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

    

###########################################################################
# Main
###########################################################################


def car_detect_init():
    global PICKLE_READY
    global svc, X_scaler

    if PICKLE_READY == False:  
        global color_space
        global orient 
        global pix_per_cell 
        global cell_per_block 
        global hog_channel 
        global spatial_size 
        global hist_size 
        global spatial_feat 
        global hist_feat 
        global hog_feat 

        print("car_detect_init......")

        # Read in cars and notcars
        cars = glob.glob('data/vehicles/*/*.png')
        notcars = glob.glob('data/non-vehicles/*/*.png')

        #just for debug
        if SMALL_DATASET == True:
            cars = cars[0:1000]
            notcars = notcars[0:1000]

        print("length of cars: ",len(cars))
        print("length of notcars: ",len(notcars))
        
        car_features = extract_features(cars)
        notcar_features = extract_features(notcars)
        
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        
        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC 
        svc = LinearSVC()
        
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        pickle.dump( svc, open( "pickle/svc.p", "wb" ))
        pickle.dump( X_scaler, open( "pickle/X_scaler.p", "wb" ))
    else:
        svc = pickle.load( open( "pickle/svc.p", "rb" ))
        X_scaler = pickle.load( open( "pickle/X_scaler.p", "rb" ))

#initializing before processing "new" video.
def car_detect_init_for_video():
    global heat_list
    global is_video
    heat_list = []
    #is_video = True   


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takesthe form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def car_detect(img):
    global y_start_stop
    global orient,pix_per_cell, cell_per_block, hog_channel
    global svc,X_scaler
    global scale
    global is_video

    ystart = y_start_stop[0]
    ystop = y_start_stop[1]
    
    draw_img = np.copy(img)

    #!!! this is not necessary because we're using cv2 functions consistantly !!!
    #img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape 
        #print("imshape", imshape)   #(256,1280,3)
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    #print("ch1.shape", ch1.shape) #(170,853)
    #print("TEST1" , nxblocks,nyblocks,nfeat_per_block) #105 20 36
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

    #1 for cells_per_step gave me a lot of improvement!!!
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    #print("TEST2", nblocks_per_window,nxsteps,nysteps)  #7 50 7
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1,feature_vec=False)
    hog2 = get_hog_features(ch2,feature_vec=False)
    hog3 = get_hog_features(ch3,feature_vec=False)
    
    #It maintains the boundary boxes that a car is detected. 
    box_list=[]

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg)
            hist_features = color_hist(subimg)

            # Scale features and make a prediction
            oo = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)    
            test_features = X_scaler.transform(oo)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #add the boundary box to box_list
                box_list.append(  ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))  )

    #1st filter
    heat = np.zeros_like(draw_img[:,:,0]).astype(np.float)
    heat = add_heat(heat,box_list)
    heat = apply_threshold(heat,2)
    heatmap =np.clip(heat,0,255)
    labels = label(heatmap)

    if is_video == False:
        #draw_img = draw_labeled_bboxes(draw_img,labels)

        x1=0
        x2=0
        y1=0
        y2=0
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            done=False

            #smoothing!!
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[1][0]
            y2 = bbox[1][1]

            #too narrow object must not be a car!!!
            #the ratio must be reaonable...!!!
            for c in cars:
                if ((x2 - x1) < 30):
                    cars.remove(c)
                elif (x2 - x1) != 0: 
                    if((y2 - y1)/(x2 - x1)) > 1.2: 
                       cars.remove(c)

            for c in cars:
                if c.distance(bbox[0],bbox[1]) < 100 :


                    a = ((x1 + c.x1)//2 , (y1 + c.y1)//2 )
                    b = ((x2 + c.x2)//2 , (y2 + c.y2)//2 )
                    cars.append(Car(a,b, c.ref+3))
                    cars.remove(c)
                    done=True
            #new one
            if done == False:
                cars.append(Car(bbox[0],bbox[1], 3))

        for c in cars:
            c.dec_ref()
            if c.ref == 0:
                cars.remove(c)

        #too narrow object must not be a car!!!
        #the ratio must be reaonable...!!!
        for c in cars:
            if ((x2 - x1) < 30):
                cars.remove(c)
            elif (x2 - x1) != 0: 
                if((y2 - y1)/(x2 - x1)) > 1.2: 
                   cars.remove(c)


        print(len(cars))

        #draw
        for c in cars:
            if c.valid() == True:
                cv2.rectangle(draw_img, c.a,c.b, (0,0,255), 6)

        return draw_img
    else:
        #if it's a video frame, we have to consider continuous frames!!!
        global heat_list
        global HEAT_LST_LIMIT 
        global valid_frames

        #trying to remain HEAT_LST_LIMIT number of heat_list.
        if len(heat_list) >= HEAT_LST_LIMIT:
           heat_list.pop(0)  #remove front one!

        heat_list.append(labels[0])

        #2st filter
        if len(heat_list) == HEAT_LST_LIMIT:
            heat = np.zeros_like(draw_img[:,:,0]).astype(np.float)    #init!
            
            for bs in heat_list:
                heat = add_heat(heat,bs)

            heat = apply_threshold(heat, valid_frames)

            heatmap =np.clip(heat,0,255)
            labels = label(heatmap)
            draw_img = draw_labeled_bboxes(draw_img,labels)
        else:
            pass  # return original image

    return draw_img

        




############################################################################################
# Project 4: Advanced Line Finding
#
# Following code came from my 4th project! 
# Most of the code is the same as before. Main difference is calling car_detect_init().
###########################################################################################

#It took some time to do Sliding window search.It will be better to avoid it as much as possible.
#we can't skip it from the beginning. For new video, it is done about the first frame.
#After than, we can skip it. However, It must be done when the lane line is not clear.
need_windowing=True

#these variables need to be defined globally for draw_laneline() 
left_fit = None
right_fit = None
left_fitx = None
right_fitx = None

#all is based on pixel. when putting text on the upper left corner, these are translated in meters.
left_curvature = 0
right_curvature =0

middle_point_off = 0.0

#It keeps the image that was drawn on a previous frame.
#When the lane lines doesn't seem to be good, this image(previous one) is used.
newwarp = None


#it was assumed that the lane is 3.7 meters wide
M_PER_PIXEL = 3.7/700



def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    if len(img.shape) ==3:
        ignore_mask_color = (255,255,255)
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#this function is for getting vertices. 
def get_vertices(shape):
    xmax = shape[1]
    ymax = shape[0]
    gap = int(xmax*0.01) #it is kind of a unit length when drawing.
    xhalf = int(xmax/2)
    yhalf = int(ymax/2)

    """
    The polygon is like this.
      /----\
     / /--\ \
    / /    \ \
   /_/      \_\

    """
    #the calculation is trying to eliminate useless area.
    vertices = np.array([[(xmax*0.05,ymax),
                          ((xhalf - gap*5), yhalf*1.2),
                          ((xhalf - gap*2), yhalf*1.2),
                          ((xhalf - gap*15), ymax),
                          ((xhalf + gap*15), ymax),
                          ((xhalf + gap*2), yhalf*1.2),
                          ((xhalf + gap*5), yhalf*1.2),
                          (xmax*0.95,ymax) ]],    dtype=np.int32)

    return vertices



#calculate the radius of curvature using pixel.
#when this value is shown on the frame, the value is translated into meter.
def check_curvature(a,b,c,X):
        yp = a*2 *X + b
        ypp = a*2
        radius_of_curvature = ((1 + yp**2)**3/2)/abs(ypp)
        #print(radius_of_curvature)

        #check if the radius is from 2000 to 30000.
        #if the value is out of this range, it's not valid. 
        #becuase ane lines can't be curved accidently in general.
        #well... actually, I got this range from trial and error.
        if radius_of_curvature  > 2000 and  radius_of_curvature  < 30000:
           return (True , radius_of_curvature)
        else:
           return (False, radius_of_curvature)


def draw_lanelines(image):
    global need_windowing
    global newwarp 

    global left_fit
    global right_fit

    global left_fitx
    global right_fitx

    global left_curvature 
    global right_curvature 

    global mtx
    global dist

    global middle_point_off

    #if finding lane line is not possible with this frame, use previous one.
    use_old_one = False

    image = cv2.undistort(image, mtx, dist, None, mtx)

    vertices = get_vertices(image.shape)
    out = region_of_interest( image , vertices)

    #draw the polygon.
    #dd = [ list(c)  for c  in vertices[0]]
    #for i in range(0,len(dd)-1):
    #    cv2.line(image, tuple(dd[i]), tuple(dd[i+1]), color=(0,0,255), thickness=5)
    #

    kernel_size = 3 #kernel size
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Take the derivative in x & y 
    xSobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    ySobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # 3) Take the absolute value of the derivative or gradient
    xAbs = np.absolute(xSobel)
    yAbs = np.absolute(ySobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    xScale = np.uint8(255*xAbs/np.max(xAbs))
    yScale = np.uint8(255*yAbs/np.max(yAbs))

    # 5) Create a mask of 1's 
    thresh=(20, 100)

    xGrad = np.zeros_like(xScale)
    xGrad[(xScale >= thresh[0]) & (xScale <= thresh[1])] = 1

    yGrad = np.zeros_like(yScale)
    yGrad[(yScale >= thresh[0]) & (yScale <= thresh[1])] = 1
    #############################################################
    # 6) Calculate the magnitude(binary)
    thresh=(30,100)
    mag = np.sqrt( xGrad**2 + yGrad **2)
    mag = np.absolute(mag)
    mag = np.uint8(255*mag/np.max(mag))

    mag_bin = np.zeros_like(mag)
    mag_bin[(mag >= thresh[0]) & (mag <= thresh[1] )] = 1
    #############################################################
    # 7) calculate the direction of the gradient
    thresh=(0.7,1.3) 
    dir_ = np.arctan2(ySobel,xSobel) 
    dir_bin = np.zeros_like(dir_)
    dir_bin [(dir_ >=  thresh[0]) & (dir_ <=  thresh[1] )] = 1

    ##############################################################
    # 8) channel in HLS
    ##############################################################
    hls = cv2.cvtColor(out, cv2.COLOR_BGR2HLS)

    #H = hls[:,:,0]
    #L = hls[:,:,1]
    S = hls[:,:,2]
    low = 150
    high = 255


    ##############################################################
    # 9) try to get the best result by combining 4 elements of upper code.
    ##############################################################
    combined = np.zeros_like(dir_bin,dtype=np.uint8)

    #this fomula gave me a good result!
    combined[((S > low) & (S <= high)) | 
             ((xGrad == 1) & (yGrad == 1) & (mag_bin == 1) & (dir_bin == 1))] = 1

    #combined[((S > low) & (S <= high)) &  (combined == 1)] = 255
    #combined[(S > low) & (S <= high)] = 255

    #just for writeup report
    if debug == True:
        cv2.imwrite('writeup_images/ombined_binary.jpg',combined)

    ###############################################################
    # 10) perspective transform
    ##############################################################

    #I got these values directly by an image viewer.
    src = np.float32( [[540,495],
                      [750,495],
                      [376,603],
                      [922,603]])
    
    #I got these values by trial & error.
    dst = np.float32( [[400,400],
                      [600,400],
                      [400,603],
                      [600,603]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(combined, M, (image.shape[1],image.shape[0]), flags = cv2.INTER_LINEAR)

    ##################################################################

    if debug == True:
        cv2.imwrite( 'writeup_images/binary_warped.jpg',warped)
        test_perspected = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags = cv2.INTER_LINEAR)
        cv2.imwrite( 'writeup_images/perspected_transform_before.jpg',image)
        cv2.imwrite( 'writeup_images/perspected_transform_after.jpg',test_perspected)

    ###########################################################
    # windowing if needed
    ###########################################################
    if need_windowing == True:
        # Assuming you have created a warped binary image called "warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

        # Create an output image to draw on and  visualize the result
        #out_img = np.dstack((warped, warped, warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)  

        #if we just use midpoint, 
        #I found out nothing was been drawing sometimes because of a white car!
        leftx_base = np.argmax(histogram[:(midpoint-200)])
        rightx_base = np.argmax(histogram[(midpoint+200):]) + midpoint

        #how much is this car off from the middle point?
        middle_point_off = (midpoint - ((leftx_base + rightx_base)/2)) * M_PER_PIXEL
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix =50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds =((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        need_windowing = False
    else:
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "warped")
        # It's now much easier to find line pixels!
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))  


    # Extract left and right line pixel positions
    leftx =  nonzerox[left_lane_inds] 
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #if there is too few pixel, it's not good. just use old one.
    if len(leftx) >  20  and len(lefty) > 20 and len(rightx) >  20  and len(righty) > 20  :
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )


        #it's based on pixcel
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        #in meters of real world 
        #left_fit = np.polyfit(lefty*ym_per_pix, leftx*M_PER_PIXEL, 2)
        #right_fit = np.polyfit(righty*ym_per_pix, rightx*M_PER_PIXEL, 2)


        isok , left_curvature = check_curvature(left_fit[0],left_fit[1],left_fit[2],300)
        if isok == True:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        else:
            need_windowing = True
            use_old_one = True
        
        isok , right_curvature = check_curvature(right_fit[0],right_fit[1],right_fit[2],300)
        if isok == True:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        else:
            need_windowing = True
            use_old_one = True  
    else:
        use_old_one = True


    # if use_old_one == True ,then use backup of newwarp
    if use_old_one == False:
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    image = car_detect(image) 

    ##############################################################
    #put some text on the frame/image
    ##############################################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = ("Left  curvature:%.3f" % (left_curvature*M_PER_PIXEL) ) + "m"
    cv2.putText(image,text ,(100,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    text = ("Right curvature:%.3f" % (right_curvature*M_PER_PIXEL) )+ "m"
    cv2.putText(image,text ,(100,150), font, 1,(255,255,255),2,cv2.LINE_AA)

    text=("from center :%.3f" % abs(middle_point_off)) + "m Left" if middle_point_off > 0 else " Right"
    cv2.putText(image,text ,(100,200), font, 1,(255,255,255),2,cv2.LINE_AA)



    #XXX newwarp = newwarp.astype(np.float32)

    #if it's not valid, just  use preview newwarp!!
    # Combine the result with the original image
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0) #alpha=1 ,beta=0.3 ,gamma=0



def main():
    global mtx,dist

    car_detect_init()  #project 5!

    # WRITEUP camera calibration  #########################################
    if PICKLE_READY == False:
        # make object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')
        
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        print("1. camera calibration - done")
        # undistortion test ########################################
        img = cv2.imread('camera_cal/calibration1.jpg')
        img_size = (img.shape[1], img.shape[0])
        
        # Do camera calibration. mtx and dist will be used for next step.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('writeup_images/calibration1_undist_before.jpg',img)
        cv2.imwrite('writeup_images/calibration1_undist_after.jpg',dst)


        # undistortion test  with test image #########################
        img = cv2.imread('test_images/test1.jpg')
        img_size = (img.shape[1], img.shape[0])
        
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('writeup_images/ndistorted_test_image.jpg',dst)
        
        #save the result as a pickle in order to save time.
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "pickle/dist_pickle.p", "wb" ) )
    else:
        #reload the pickle image
        dist_pickle = {}
        dist_pickle =  pickle.load(open( "pickle/dist_pickle.p", "rb" ) )
        mtx = dist_pickle["mtx"]  
        dist = dist_pickle["dist"] 

    print("2. undistortion test - done")

    ## WRITEUP Pipeline(Single Images)  ##########################
    for f in os.listdir("test_images/"):
        global need_windows

        image = cv2.imread('test_images/' + f )
        need_windowing=True
        out  = draw_lanelines(image)
        cv2.imwrite( 'output_images/' + f ,out)

        #skip some debug code from now on. 
        debug=False 
        #sys.exit(1)
    print("3. single images pipeline - done")
        
    ## WRITEUP Pileline(Video) ###################################

    need_windowing=True
    clip = VideoFileClip("test_video.mp4")
    car_detect_init_for_video()
    result = clip.fl_image(draw_lanelines) 
    result.write_videofile('test_video_output.mp4', audio=False)
    print("4. video pipeline - done")

    need_windowing=True
    clip = VideoFileClip("project_video.mp4")
    car_detect_init_for_video()
    result = clip.fl_image(draw_lanelines) 
    result.write_videofile('project_video_output.mp4', audio=False)
    print("4. video pipeline - done")


if __name__ == "__main__":
    main()
