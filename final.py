#######################################################################
# Udacity Self Driving Car Nanodegree 
#
# Project 5: Vehicle Detection
# by Ghicheon Lee 
#
# date: 2018.2.17
#######################################################################

#I referenced a lot of codes and hints from "Vehicle Detection" Lecture of CarND!


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
import os 
import sys

from one import car_detect_init,car_detect

#avoid doing carmera callibration and getting distortion parameter values again.
PICKLE_READY = False
#PICKLE_READY = True

#just for debug & writeup report.
debug=False
#debug=True


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

##########################################
car_detect_init()


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


def draw_lanelines(image, is_video = False):
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
        cv2.imwrite('writeup_combined_binary.jpg',combined)

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
        cv2.imwrite( 'writeup_binary_warped.jpg',warped)
        test_perspected = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags = cv2.INTER_LINEAR)
        cv2.imwrite( 'writeup_perspected_transform_before.jpg',image)
        cv2.imwrite( 'writeup_perspected_transform_after.jpg',test_perspected)

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

        #cv2.circle(color_warp,(600,400), 30, (255,0,0), -1) #just for debug


        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    image = car_detect(image , is_video) #XXXXXXXx !!!!!!!!!!!!!!

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

###########################################################################
#main code
###########################################################################



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
    
            #Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)
    
    #cv2.destroyAllWindows()
    print("1. camera calibration - done")
    

# undistortion test ########################################
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration. mtx and dist will be used for next step.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('writeup_calibration1_undist_before.jpg',img)
    cv2.imwrite('writeup_calibration1_undist_after.jpg',dst)


# undistortion test  with test image #########################
    img = cv2.imread('test_images/test1.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('writeup_undistorted_test_image.jpg',dst)
    
    #save the result as a pickle in order to save time.
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "dist_pickle.p", "wb" ) )
else:
    #reload the pickle image
    dist_pickle = {}
    dist_pickle =  pickle.load(open( "dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]  
    dist = dist_pickle["dist"] 

print("2. undistortion test - done")

## WRITEUP Pipeline(Single Images)  ##########################
for f in os.listdir("test_images/"):
    global need_windows

    image = cv2.imread('test_images/' + f )
    need_windowing=True
    out  = draw_lanelines(image)
    cv2.imwrite( 'test_images_output/' + f ,out)

    #skip some debug code from now on. 
    debug=False 
    #sys.exit(1)
print("3. single images pipeline - done")
    
## WRITEUP Pileline(Video) ###################################

need_windowing=True
clip = VideoFileClip("test_video.mp4")
result = clip.fl_image(lambda x:draw_lanelines(x,True)) 
result.write_videofile('test_video_output.mp4', audio=False)
print("4. video pipeline - done")

#need_windowing=True
#clip = VideoFileClip("project_video.mp4")
#result = clip.fl_image(draw_lanelines) 
#result.write_videofile('project_video_output.mp4', audio=False)
#print("4. video pipeline - done")
#
