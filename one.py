import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import pickle
from scipy.ndimage.measurements import label


### TODO: Tweak these parameters and see how the results change.
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
svc = None
X_scaler = None


#how many continuous frames do we maintain for calculating "car detection"?
HEAT_LST_LIMIT = 10

#if cars are detected on valid_frames continuous  frames, the detections is considered valid.
#valid_frames=9   
#valid_frames=12  
valid_frames=7

#used in car_detect().
scale = 1.5


PICKLE_USE = True
#PICKLE_USE = False

SMALL_DATASET = False
#SMALL_DATASET = True

is_video = False    #!!!

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
    global PICKLE_USE


    if PICKLE_USE == False:  #XXXXXXXXXXXX
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
        global svc 
        global X_scaler 

        print("car_detect_init......")

        # Read in cars and notcars
        cars = glob.glob('vehicles/*/*.png')
        notcars = glob.glob('non-vehicles/*/*.png')

        if SMALL_DATASET == True:
            cars = cars[0:1000]
            notcars = notcars[0:1000]

        #cars = glob.glob('v/*.png')
        #notcars = glob.glob('n/*.png')


        print(len(cars))
        print(len(notcars))
        
        
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

        pickle.dump( svc, open( "svc.p", "wb" ))
        pickle.dump( X_scaler, open( "X_scaler.p", "wb" ))
        #print("X_train0 ", X_train[0])
    else:
        global svc
        global X_scaler
        svc = pickle.load( open( "svc.p", "rb" ))
        X_scaler = pickle.load( open( "X_scaler.p", "rb" ))

#initializing before processing new video.
def car_detect_init_for_video():
    global heat_list
    heat_list = []
    is_video = True    #!!!


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takesthe form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
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


################################################################################################


# maintain some continuous frames. 
heat_list = []

# Define a single function that can extract features using hog sub-sampling and make predictions
def car_detect(img):
    global y_start_stop
    global orient,pix_per_cell, cell_per_block, hog_channel
    global svc,X_scaler
    global scale

    ystart = y_start_stop[0]
    ystop = y_start_stop[1]
    
    draw_img = np.copy(img)

    #img = img.astype(np.float32)/255
    #img = img.astype(np.float32)
    #print(img[0,0,0])
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape 
        print("imshape", imshape)   #(256,1280,3)
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    print("ch1.shape", ch1.shape) #(170,853)
    print("A", nxblocks,nyblocks,nfeat_per_block) #105 20 36
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

    #XXXXXXXXXXXXXXXXXXXXXXXXXXXX
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    print("A", nblocks_per_window,nxsteps,nysteps)  #7 50 7
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1,feature_vec=False)
    hog2 = get_hog_features(ch2,feature_vec=False)
    hog3 = get_hog_features(ch3,feature_vec=False)
    
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
                box_list.append(  ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))  )


    #1st filter
    heat = np.zeros_like(draw_img[:,:,0]).astype(np.float)
    heat = add_heat(heat,box_list)
    heat = apply_threshold(heat,2)
    heatmap =np.clip(heat,0,255)
    out1 = label(heatmap)

    global is_video

    if is_video == False:
        draw_img = draw_labeled_bboxes(draw_img,out1)
        return draw_img
    else:
        global heat_list
        global HEAT_LST_LIMIT 
        global valid_frames
        if len(heat_list) >= HEAT_LST_LIMIT:
           heat_list.pop(0)  #remove front one.

        heat_list.append(out1[0])

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

        

