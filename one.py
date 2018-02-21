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
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
svc = None
X_scaler = None

#PICKLE_USE = True
PICKLE_USE = False

#################################################################################

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

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgs, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    global color_space
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        image = cv2.imread(file)

        img_features= single_img_features( image, spatial_size,
                                           hist_bins, orient, 
                                           pix_per_cell, cell_per_block, hog_channel,
                                           spatial_feat, hist_feat, hog_feat)

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
def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    global color_space
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img)
    #3) Compute spatial features if flag is set

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #print("TRAIN1...", type(np.array(hog_features)) , type(spatial_features) , type(hist_features))
    #print("TRAIN2...", (np.array(hog_features)).shape , (spatial_features).shape , (hist_features).shape)
    #print("TRAIN3...", hog_features[0] , hog_features[1] , hog_features[2] )
    #print("TRAIN4..." , spatial_features[0] , hist_features[0])


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
        global hist_bins 
        global spatial_feat 
        global hist_feat 
        global hog_feat 
        global svc 
        global X_scaler 

        print("car_detect_init......")

        # Read in cars and notcars
        cars = glob.glob('vehicles/*/*.png')
        notcars = glob.glob('non-vehicles/*/*.png')

        #cars = glob.glob('v/*.png')
        #notcars = glob.glob('n/*.png')


        print(len(cars))
        print(len(notcars))
        
        
        car_features = extract_features(cars, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(notcars, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        
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


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
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



# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    #print(draw_img)
    #print(draw_img.shape)

    img = img.astype(np.float32)/255
    #print(img)
    #print(img.shape)
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
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
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            #print("PREDICT1...", type(hog_features) , type(spatial_features) , type(hist_features))
            #print("PREDICT2...", hog_features.shape , (spatial_features).shape , (hist_features).shape)
            #print("PREDICT3...", hog_features[0] , hog_features[1] , hog_features[2] )
            #print("PREDICT4..." , spatial_features[0] , hist_features[0])


            # Scale features and make a prediction
            oo = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)    
            test_features = X_scaler.transform(oo)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img

def car_detect(img, is_video = True):
    global color_space
    global orient 
    global pix_per_cell 
    global cell_per_block 
    global hog_channel 
    global spatial_size 
    global hist_bins 
    global spatial_feat 
    global hist_feat 
    global hog_feat 
    global y_start_stop 
    global svc 
    global X_scaler

    ystart = 400
    ystop = 656
    scale = 1.5
        
    out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    return out_img

#################################################################################
#################################################################################
# Define a function to return HOG features and visualization
## def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
##                         vis=False, feature_vec=True):
##     # Call with two outputs if vis==True
##     if vis == True:
##         features, hog_image = hog(img, orientations=orient, 
##                                   pixels_per_cell=(pix_per_cell, pix_per_cell),
##                                   cells_per_block=(cell_per_block, cell_per_block), 
##                                   transform_sqrt=True, 
##                                   visualise=vis, feature_vector=feature_vec)
##         return features, hog_image
##     # Otherwise call with one output
##     else:      
##         features = hog(img, orientations=orient, 
##                        pixels_per_cell=(pix_per_cell, pix_per_cell),
##                        cells_per_block=(cell_per_block, cell_per_block), 
##                        transform_sqrt=True, 
##                        visualise=vis, feature_vector=feature_vec)
##         return features
## 
## # Define a function to compute binned color features  
## def bin_spatial(img, size=(32, 32)):
##     # Use cv2.resize().ravel() to create the feature vector
##     features = cv2.resize(img, size).ravel() 
##     # Return the feature vector
##     return features
## 
## # Define a function to compute color histogram features 
## # NEED TO CHANGE bins_range if reading .png files with mpimg!
## def color_hist(img, nbins=32, bins_range=(0, 256)):
##     # Compute the histogram of the color channels separately
##     channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
##     channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
##     channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
##     # Concatenate the histograms into a single feature vector
##     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
##     # Return the individual histograms, bin_centers and feature vector
##     return hist_features
## 
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()

#def car_detect(img, is_video = True):
#    global color_space
#    global orient 
#    global pix_per_cell 
#    global cell_per_block 
#    global hog_channel 
#    global spatial_size 
#    global hist_bins 
#    global spatial_feat 
#    global hist_feat 
#    global hog_feat 
#    global y_start_stop 
#    global svc 
#    global X_scaler
#
#
#    print("car_detect......")
#
#    image = np.copy(img)
#    
#    #DELETE XXXXXXXXXXXX
#    # Uncomment the following line if you extracted training
#    # data from .png images (scaled 0 to 1 by mpimg) and the
#    # image you are searching is a .jpg (scaled 0 to 255)
#    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#    #image = image.astype(np.float32)/255
#    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#    
#    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                        xy_window=(128, 128), xy_overlap=(0.80, 0.80))
#    
#    box_list = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
#                            spatial_size=spatial_size, hist_bins=hist_bins, 
#                            orient=orient, pix_per_cell=pix_per_cell, 
#                            cell_per_block=cell_per_block, 
#                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                            hist_feat=hist_feat, hog_feat=hog_feat)                       
#    
#    ### pickle.dump( box_list, open( "bbox_pickle.p", "wb" ))
#    ### window_img = draw_boxes(draw_image, box_list, color=(0, 0, 255), thick=6)    
#    ### plt.imshow(window_img)
#    ### plt.show()
#    ### ##############################################################################
#    ### # Read in a pickle file with bboxes saved
#    ### # Each item in the "all_bboxes" list will contain a 
#    ### # list of boxes for one of the images shown above
#    ### box_list = pickle.load( open( "bbox_pickle.p", "rb" ))
#    ### 
#    
#    ####  Read in image similar to one shown above 
#    heat = np.zeros_like(image[:,:,0]).astype(np.float)
#    
#    
#    # Add heat to each box in box list
#    heat = add_heat(heat,box_list)
#        
#    # Apply threshold to help remove false positives
#    heat = apply_threshold(heat,5)
#    
#    # Visualize the heatmap when displaying    
#    heatmap = np.clip(heat, 0, 255)
#    
#    # Find final boxes from heatmap using label function
#    labels = label(heatmap)
#    draw_img = draw_labeled_bboxes(np.copy(image), labels)
#
#    return draw_img
#
    
    #fig = plt.figure()
    #plt.subplot(121)
    #plt.imshow(draw_img)
    #plt.title('Car Positions')
    #plt.subplot(122)
    #plt.imshow(draw_img)
    #plt.title('Car Positions')
    #plt.subplot(122)
    #plt.show()



################################################################################################
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
#def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
#                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#    # If x and/or y start/stop positions not defined, set to image size
#    if x_start_stop[0] == None:
#        x_start_stop[0] = 0
#    if x_start_stop[1] == None:
#        x_start_stop[1] = img.shape[1]
#    if y_start_stop[0] == None:
#        y_start_stop[0] = int(img.shape[0] * 0.4)
#    if y_start_stop[1] == None:
#        y_start_stop[1] = img.shape[0]
#    # Compute the span of the region to be searched    
#    xspan = x_start_stop[1] - x_start_stop[0]
#    yspan = y_start_stop[1] - y_start_stop[0]
#    # Compute the number of pixels per step in x/y
#    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
#    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
#    # Compute the number of windows in x/y
#    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
#    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
#    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
#    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
#    # Initialize a list to append window positions to
#    window_list = []
#    # Loop through finding x and y window positions
#    # Note: you could vectorize this step, but in practice
#    # you'll be considering windows one by one with your
#    # classifier, so looping makes sense
#    for ys in range(ny_windows):
#        for xs in range(nx_windows):
#            # Calculate window position
#            startx = xs*nx_pix_per_step + x_start_stop[0]
#            endx = startx + xy_window[0]
#            starty = ys*ny_pix_per_step + y_start_stop[0]
#            endy = starty + xy_window[1]
#            
#            # Append window position to list
#            window_list.append(((startx, starty), (endx, endy)))
#    # Return the list of windows
#    return window_list
#
#
### Define a function you will pass an image 
## and the list of windows to be searched (output of slide_windows())
#def search_windows(img, windows, clf, scaler, color_space='RGB', 
#                    spatial_size=(32, 32), hist_bins=32, 
#                    hist_range=(0, 256), orient=9, 
#                    pix_per_cell=8, cell_per_block=2, 
#                    hog_channel=0, spatial_feat=True, 
#                    hist_feat=True, hog_feat=True):
#
#    #1) Create an empty list to receive positive detection windows
#    on_windows = []
#    #2) Iterate over all windows in the list
#    for window in windows:
#        #3) Extract the test window from original image
#        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
#        #4) Extract features for that window using single_img_features()
#        features = single_img_features(test_img, color_space=color_space, 
#                                       spatial_size=spatial_size, hist_bins=hist_bins, 
#                                       orient=orient, pix_per_cell=pix_per_cell, 
#                                       cell_per_block=cell_per_block, 
#                                       hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                                       hist_feat=hist_feat, hog_feat=hog_feat)
#        #5) Scale extracted features to be fed to classifier
#        test_features = scaler.transform(np.array(features).reshape(1, -1))
#        #6) Predict using your classifier
#        prediction = clf.predict(test_features)
#        #7) If positive (prediction == 1) then save the window
#        if prediction == 1:
#            on_windows.append(window)
#    #8) Return windows for positive detections
#    return on_windows
