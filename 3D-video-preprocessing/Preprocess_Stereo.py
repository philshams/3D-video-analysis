'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                Display and Analyze a saved stereoscopic video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import cv2; import os
from depth_funcs import create_global_matchers, get_background_mean, make_striped_background, get_offset, get_biggest_contour
from depth_funcs import flip_mouse, correct_flip, write_videos


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------               Select video file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
#file_name = 'Chronic_Mantis_stim-default-645365-video-0.avi'
#file_loc = 'C:\Drive\Video Analysis\data\\'
#date = '28.02.2018\\'
#mouse_session = '205_2a\\'
#save_vid_name = 'analyze_full' # name-tag to be associated with all saved files

file_name = 'Depth_Camera-default-491687-video-0.avi' # new camera // toward door // left
file_name_2 = 'Depth_Camera-default-491687-2video-0.avi'# old camera // toward back // right

file_loc = 'C:\Drive\Video Analysis\data\\'
date = '15.03.2018\\'
mouse_session = 'bj141p2\\'
save_vid_name = 'rectified_norm' # name-tag to be associated with all saved files
two_videos = True
video_type = '2D'

undistort = True


#file_name = 'mouse0.avi'
#file_loc = 'C:\Drive\Video Analysis\data\\'
#date = '05.02.2018\\'
#mouse_session = '202-1a\\'
#save_vid_name = 'analyze_2D_test' # name-tag to be associated with all saved files



# load video
file_loc = file_loc + date + mouse_session
file = file_loc + file_name
if two_videos:
    file_2 = file_loc + file_name_2
print(file + ' loaded')
vid = cv2.VideoCapture(file)
if two_videos:
    vid2 = cv2.VideoCapture(file_2)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  


# --------------------
# Set video parameters
# --------------------
frame_rate = 40 #aesthetic frame rate of saved videos
display_frame_rate = 1000 #1000 makes it as fast as possible
start_frame = 0
end_frame = 2005 #set as either desired end frame or np.inf to go to end of movie


# ----------------------------
# Set mouse contour parameters
# ----------------------------
mask_thresh = .42 #mouse mask threshold (lower is more stringently darker than background)
kernel = [4,3] #erosion and dilation kernel sizes for mouse mask
iters = [0,7] #number of erosion and dilation iterations for mouse mask


# -----------------------------
# Set depth analysis parameters
# -----------------------------
smooth_factor = 12
final_smoothing_kernel_width = 21
stereo_axis = 'y' #axis of the two cameras with respect to the video
use_norm_to_analyze = True
zero_mean = True

time_filter_width = 2 # 0, 1 or 2
time_filter_weight = .4 # 0 to 1

window_size = 1
min_disparity = 24 #ranges from 24 to 28
num_disparities = 2*16  
max_pixel_diff = 8

pre_filter_cap = 61 
unique_ratio = 10


# ------------------------------------------
# Set mouse orientation detection parameters
# ------------------------------------------
wispy_thresh = 2 #1.15, 1.5
wispy_erosions = 8 #12
speed_thresh = 7
width_thresh= 2 #1.2


# -----------------------
# Set data-saving options
# -----------------------
show_images = True
save_data = True
write_images = False
do_not_overwrite = False

save_2D_data = True #saved cropeed video of mouse

if write_images:
    write_normal_video = False
    write_normalized_video = True
    write_cropped_mice = False
    write_stereo_inputs = False
    write_3D_combined = False
    write_3D_smooth = False




#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Inititialization -- Functions to run once per video                 -----------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -----------------------
# Perform initialization?
# -----------------------
    
initial_run = False

if initial_run:
    do_get_background = True #for background subtraction
    do_get_offsets = True #in lieu of image calibration
else:
    do_get_background = False
    do_get_offsets = False



# ----------------------------------------
# get or load background subtraction image
# ----------------------------------------

if do_get_background:
    background_mat = get_background_mean(vid, vid2, two_videos, file_loc = file_loc, avg_over_approx = 1000)
else:
    background_file_name = file_loc + 'background_mat_avg.npy'
    background_mat = np.load(background_file_name)

# ------------------------------------------------------------------------------------------------------------------------
# get or load y-offset of cameras (cheap imitation of an calibration) and x-offset (pixel disparity across stereo cameras)
# ------------------------------------------------------------------------------------------------------------------------
    
if do_get_offsets:
    y_offset, _, x_offset, _ = get_offset(vid,vid2,two_videos,background_mat,start_frame, stop_frame = start_frame+1000, l=1, r=2, mask_thresh = mask_thresh, kernel = kernel, iters = iters) #kernel and iters for erosion and dilation, respectively
    if stereo_axis == 'x':
        np.save(file_loc + 'offset.npy',y_offset)
    elif stereo_axis == 'y':
        np.save(file_loc + 'offset.npy',x_offset)
else:
    offset_file_name = file_loc + 'offset.npy'
    offset = int(np.load(offset_file_name))





#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Set up analysis                -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -------------------------------------
# Create stereoscopic 3D-image creator
# -------------------------------------
stereo_left, stereo_right, min_disparity, shift_pixels = create_global_matchers(window_size = window_size, min_disparity =min_disparity, 
                                                                                num_disparities = num_disparities, smooth_factor = smooth_factor,
                                                                                 pre_filter_cap = pre_filter_cap, unique_ratio = unique_ratio,
                                                                                 max_pixel_diff = max_pixel_diff)


# ----------------------
# Set up analysis videos
# ----------------------
crop_size = 150
square_size = (crop_size,crop_size)
border_size = 150
l = 1   #left cam is input 1, right cam is input 2...for now
r = 2
end_frame = min(end_frame, vid.get(cv2.CAP_PROP_FRAME_COUNT)-5)
if undistort == True:
    maps = np.load(file_loc + 'fisheye_maps.npy')
    map1 = maps[:,:,0:2]
    map2 = maps[:,:,2]

# make striped background # in size of ROI for 3D mouse detection -# height: crop_size+border_size and length: crop_size+2*border_size
stripesL, stripesR = make_striped_background(height, width, min_disparity, roi_height=crop_size + border_size, roi_width= crop_size + 2*border_size)

# initialize mouse video
#vid = cv2.VideoCapture(file_name)
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
if two_videos:
    vid2.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

# create background for subtraction
background_L = background_mat[:,:,0]
background_R = background_mat[:,:,1]

# initialize time-smoothing
if time_filter_width > 0:
    stereo_image_smoothed_prev4 = np.zeros((square_size)).astype(np.uint8)
    stereo_image_smoothed_prev3 = np.zeros((square_size)).astype(np.uint8)
    stereo_image_smoothed_prev2 = np.zeros((square_size)).astype(np.uint8)
    stereo_image_smoothed_prev = np.zeros((square_size)).astype(np.uint8)
    stereo_image_smoothed_cur = np.zeros((square_size)).astype(np.uint8)

# initialize mouse orientation detection
face_left = -1 #1 or -1 : initial orientation
topright_or_botleft = 1
topright_or_botleft_prev = 1
depth_ratio = np.ones(3)
cxL = 0
cyL = 0
move_prev = 0
history_x = np.zeros(4)
history_y = np.zeros(4)
ellipse = 0
ellipse_width = 0
slope_recipr = 1
disruption = 1

#initialize erosion/dilation kernels
kernel_er = np.ones((kernel[0],kernel[0]),np.uint8)
kernel_dil = np.ones((kernel[1],kernel[1]),np.uint8)
kernel_head = np.ones((9,9),np.uint8)


# ----------------------
# Select videos to save 
# ----------------------

file_loc = file_loc + save_vid_name
fourcc = cv2.VideoWriter_fourcc(*'XVID') #LJPG for lossless, XVID or MJPG works for compressed

# write videos as selected above
if write_images:
    normal_video, normalized_video, cropped_mouse, stereo_input_L, stereo_input_R, threeD_combined, threeD_smooth = \
                    write_videos(file_loc, write_images, do_not_overwrite, fourcc, frame_rate, width, height, crop_size, border_size,
                     write_normal_video, write_normalized_video, write_cropped_mice, write_stereo_inputs, write_3D_combined, write_3D_smooth)

# save data for further analysis
if video_type=='2D':
    save_2D_data = True
if save_data == True:
    fourcc = cv2.VideoWriter_fourcc(*'LJPG') #LJPG for lossless
    data_file = file_loc + '_data.avi'
    if os.path.isfile(data_file) and do_not_overwrite:
        raise Exception('File already exists') 
    if save_2D_data and video_type=='2D':
        data_video_2D = cv2.VideoWriter(data_file,fourcc , frame_rate, (crop_size,crop_size), False) 
    elif save_2D_data and not video_type=='2D':
        data_video_3D = cv2.VideoWriter(data_file,fourcc , frame_rate, (crop_size,crop_size), False) 
        data_video_2D = cv2.VideoWriter(file_loc + '_2D_data.avi',fourcc , frame_rate, (crop_size,crop_size), False) 
    data_times = np.array([])
    mouse_coordinates = []
    mouse_velocity = []
    disruptions = []
 
    

   
          
#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Perform analysis                ------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


while True:
    # grab the frame
    ret, frame = vid.read()
    frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES) 
    if two_videos:
        ret2, frame2 = vid2.read()

    if ret and ret2:  

        # ------------------------------
        # Perform background subtraction
        # -------------------------------
        # separate left and right frames
        frame_L = frame[:,:,l] 
        
        if two_videos:
            frame_R = frame2[:,:,l] #two separate videos
        elif video_type == '2D':
            frame_R = np.zeros(frame_L.shape) #just use one camera
        else:
            frame_R = frame[:,:,r] #stereoscopic camera
    
        
        # divide image by the average background to emphasize the mouse
        frame_norm_L = (256/2 * frame_L / background_L ).astype(np.uint8)
        frame_norm_R = (256/2 * frame_R / background_R ).astype(np.uint8)
        
        #rectify image
        if undistort == True:
            frame_norm_L = cv2.remap(frame_norm_L, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            frame_norm_R = cv2.remap(frame_norm_R, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
          
        # ----------------------------------
        # Find the contour around the mouse
        # ----------------------------------
        
        # use the thresholds, erosion, and dilation set above to extract a mask coinciding with the mouse        
        frame_norm_L_mask = ((frame_norm_L / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=iters[0]) 
        frame_norm_L_mask = cv2.dilate(frame_norm_L_mask, kernel_dil, iterations=iters[1]) 

        # extract the largest contour in this mask -- should correspond to the mouse
        cxL_prev = cxL #first, save the previous center of mouse in order to extract velocity
        cyL_prev = cyL #first, save the previous center of mouse in order to extract velocity
        try:
            contoursL, big_cnt_indL, cxL, cyL, cntL = get_biggest_contour(frame_norm_L_mask)
        except:
            print('mouse out of bounds at y = ' + str(cyL))
            disruption = 1
            continue
            
        # create a new mask, corresponding to only the largest contour
        blank = np.zeros(frame_norm_L.shape).astype(np.uint8)
        contour_mask_L = cv2.drawContours(blank, contoursL, big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED)
        
        # apply this mask to the original image
        if use_norm_to_analyze:
            frame_L_masked = frame_norm_L * contour_mask_L
        else:
            if undistort == True:
                frame_L = cv2.remap(frame_L, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                frame_R = cv2.remap(frame_R, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                
            frame_L_masked = frame_L * contour_mask_L
        
        
        # crop the original, masked image to the crop size
        frame_L_masked_padded = cv2.copyMakeBorder(frame_L_masked,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
        if video_type=='2D':
            frame_L_masked_cropped = frame_L_masked_padded[cyL-int(crop_size/2)+border_size:cyL+int(crop_size/2)+border_size,cxL-int(crop_size/2)+border_size:cxL+int(crop_size/2)+border_size]    
            

        if not video_type == '2D': 
            
            # get the image to the cropped size
            contour_mask_L_padded = cv2.copyMakeBorder(contour_mask_L,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            reduced_contour_mask_L = contour_mask_L_padded[cyL+int(border_size/2-crop_size/2):cyL+int(3*border_size/2 + crop_size/2),cxL-border_size:cxL+crop_size + border_size]
            
            #do all of the above for the other frame
            frame_norm_R_mask = ((frame_norm_R / (256/2)) < mask_thresh).astype(np.uint8) 
            frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=iters[0])
            frame_norm_R_mask = cv2.dilate(frame_norm_R_mask, kernel_dil, iterations=iters[1])
            
            contoursR, big_cnt_indR, cxR, cyR, cntR = get_biggest_contour(frame_norm_R_mask)
            #print(cyR - cyL)
            stripesL, stripesR = make_striped_background(height, width, (cyR-cyL), roi_height=crop_size + border_size, roi_width= crop_size + 2*border_size)

            
            blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
            contour_mask_R = (cv2.drawContours(blank, contoursR, big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
            if use_norm_to_analyze:
                frame_R_masked = frame_norm_R * contour_mask_R
            else:
                frame_R_masked = frame_R * contour_mask_R
                
            
            contour_mask_R_padded = cv2.copyMakeBorder(contour_mask_R,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            frame_R_masked_padded = cv2.copyMakeBorder(frame_R_masked,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            
            
            
            #if stereo cameras along y-axis, flip image
            if stereo_axis=='y':
                frame_L_masked_for_stripes = frame_L_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxL-border_size:cxL+crop_size + border_size]
                frame_R_masked_for_stripes = frame_R_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxR-border_size:cxR+crop_size + border_size]
                reduced_contour_mask_R = contour_mask_R_padded[cyL+int(border_size/2-crop_size/2):cyL+int(3*border_size/2 + crop_size/2),cxR-border_size:cxR+crop_size + border_size]
                
                M = cv2.getRotationMatrix2D((reduced_contour_mask_L.shape[1]/2,reduced_contour_mask_L.shape[0]/2),-90,1)
                reduced_contour_mask_L = cv2.warpAffine(reduced_contour_mask_L,M,(reduced_contour_mask_L.shape[1],reduced_contour_mask_L.shape[0]))
                reduced_contour_mask_R = cv2.warpAffine(reduced_contour_mask_R,M,(reduced_contour_mask_R.shape[1],reduced_contour_mask_R.shape[0]))
                M = cv2.getRotationMatrix2D((frame_L_masked_for_stripes.shape[1]/2,frame_L_masked_for_stripes.shape[0]/2),-90,1)
                frame_L_masked_for_stripes = cv2.warpAffine(frame_L_masked_for_stripes,M,(frame_L_masked_for_stripes.shape[1],frame_L_masked_for_stripes.shape[0]))
                frame_R_masked_for_stripes = cv2.warpAffine(frame_R_masked_for_stripes,M,(frame_R_masked_for_stripes.shape[1],frame_R_masked_for_stripes.shape[0]))
            elif stereo_axis=='x':
                frame_L_masked_for_stripes = frame_L_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxL-border_size:cxL+crop_size + border_size]
                frame_R_masked_for_stripes = frame_R_masked_padded[cyR+border_size-int(stripesL.shape[0]/2):cyR+border_size+int(stripesL.shape[0]/2),cxL-border_size:cxL+crop_size + border_size]
                reduced_contour_mask_R = contour_mask_R_padded[cyR+int(border_size/2-crop_size/2):cyR+int(3*border_size/2 + crop_size/2),cxL-border_size:cxL+crop_size + border_size]
                
            # create a striped mask of size set above, in which to place the image of the mouse that was just extracted 
            stripe_maskL = stripesL * (1 - reduced_contour_mask_L)
            frame_L_masked_striped =  frame_L_masked_for_stripes + stripe_maskL
               
            stripe_maskR = stripesR * (1 - reduced_contour_mask_R)
            frame_R_masked_striped = frame_R_masked_for_stripes + stripe_maskR
            
            #also crop the masked image to the crop size
            frame_L_masked_cropped = frame_L_masked_padded[cyL-int(crop_size/2)+border_size:cyL+int(crop_size/2)+border_size,cxL-int(crop_size/2)+border_size:cxL+int(crop_size/2)+border_size] 
            frame_R_masked_cropped = frame_R_masked_padded[cyR-int(crop_size/2)+border_size:cyR+int(crop_size/2)+border_size,cxR-int(crop_size/2)+border_size:cxR+int(crop_size/2)+border_size]

        # --------------------------
        # Compute 3D image of mouse
        # --------------------------    
                        
            # compute stereo images from each direction
            stereo_image_L = stereo_left.compute(frame_L_masked_striped,frame_R_masked_striped).astype(np.uint8)
            stereo_image_R = stereo_right.compute(frame_R_masked_striped,frame_L_masked_striped).astype(np.uint8)
            
            #if stereo cameras along y-axis, unflip image
            if stereo_axis=='y':

                M = cv2.getRotationMatrix2D((stereo_image_L.shape[1]/2,stereo_image_L.shape[0]/2),90,1)
                stereo_image_L = cv2.warpAffine(stereo_image_L,M,(stereo_image_L.shape[1],stereo_image_L.shape[0]))
                stereo_image_R = cv2.warpAffine(stereo_image_R,M,(stereo_image_R.shape[1],stereo_image_R.shape[0]))
                                               
                stereo_image_L_cropped = stereo_image_L[int(stereo_image_L.shape[0]/2 - crop_size/2):int(stereo_image_L.shape[0]/2 + crop_size/2), int(2*border_size - crop_size/2):int(2*border_size + crop_size/2)]
                stereo_image_R_cropped = stereo_image_R[int(stereo_image_L.shape[0]/2 - crop_size/2)-(cyL-cyR):int(stereo_image_L.shape[0]/2 + crop_size/2)-(cyL-cyR), int(2*border_size - crop_size/2):int(2*border_size + crop_size/2)]

            else:
                # crop the stereo images to the crop size set above
                stereo_image_L_cropped = stereo_image_L[int(border_size/2):int(border_size/2+crop_size), int(2*border_size - crop_size/2):int(2*border_size + crop_size/2)]
                stereo_image_R_cropped = stereo_image_R[int(border_size/2):int(border_size/2+crop_size), int(2*border_size - crop_size/2)-(cxL-cxR):int(2*border_size + crop_size/2)-(cxL-cxR)]              
                
            # combine the two stereo images into an average, on the pixels on which both cameras agree there was a mouse
            stereo_image_combined = ((stereo_image_L_cropped.astype(uint16) + (255 - stereo_image_R_cropped.astype(uint16))) / 2 *(frame_R_masked_cropped>0) *(frame_L_masked_cropped>0)).astype(np.uint8)
            if zero_mean:
                zero_mean_array = stereo_image_combined[stereo_image_combined>0] / np.mean(stereo_image_combined[stereo_image_combined>0])
                zero_mean_array[zero_mean_array > 2] = 2
                stereo_image_combined[stereo_image_combined>0] = (zero_mean_array * 128 -1).astype(uint8)
            #print(np.mean( stereo_image_combined[stereo_image_combined>0] ))

            
        # ------------------------
        # Smooth 3D image of mouse
        # ------------------------
    
            # store previous smoothed stereo images in order to smooth this image over time
            if time_filter_width > 0:
                stereo_image_smoothed_prev4 = stereo_image_smoothed_prev3
                stereo_image_smoothed_prev3 = stereo_image_smoothed_prev2
                stereo_image_smoothed_prev2 = stereo_image_smoothed_prev
                stereo_image_smoothed_prev = stereo_image_smoothed_cur
            
            # do Gaussian smoothing on the current, combined stereo image
            stereo_image_smoothed_cur = cv2.GaussianBlur(stereo_image_combined,ksize=(final_smoothing_kernel_width,final_smoothing_kernel_width),
                                                                   sigmaX=final_smoothing_kernel_width,sigmaY=final_smoothing_kernel_width)
            
            # do temporal smoothing over the past 3 or 5 frames
            if frame_num < start_frame + 1 +2*time_filter_width and time_filter_width > 0:
                continue
            elif time_filter_width == 0:
                stereo_image_smoothed = stereo_image_smoothed_cur
            elif time_filter_width == 1:
                stereo_image_smoothed = (np.sum(np.array([time_filter_weight*stereo_image_smoothed_cur,stereo_image_smoothed_prev,
                                                                        time_filter_weight*stereo_image_smoothed_prev2]),axis=0)/(1+2*time_filter_weight)).astype(np.uint8)
            elif time_filter_width == 2:
                stereo_image_smoothed = (np.sum(np.array([time_filter_weight*time_filter_weight*stereo_image_smoothed_cur,time_filter_weight*stereo_image_smoothed_prev,
                                                                        stereo_image_smoothed_prev2,time_filter_weight*stereo_image_smoothed_prev3,
                                                                        time_filter_weight*time_filter_weight*stereo_image_smoothed_prev4]),
                                                                        axis=0)/(1+2*time_filter_weight+2*time_filter_weight*time_filter_weight)).astype(np.uint8)   

            
        # --------------------------
        # Get speed and orientation
        # --------------------------
        # flip mouse into the correct orientation
        try:
            rotate_angle, face_left, ellipse, topright_or_botleft, ellipse_width = \
            flip_mouse(face_left, ellipse, topright_or_botleft,frame_L_masked_cropped , sausage_thresh = 1.1)
        except:
            print('mouse out of bounds at y = ' + str(cyL))
            disruption = 1
            continue


        M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1) 
        frame_L_masked_straight = cv2.warpAffine(frame_L_masked_cropped,M,(crop_size,crop_size))        

        # check for errors -- if the tail end is less tumescent, or the mouse is running toward its tail, slip 180 degrees
        frame_L_eroded = cv2.erode(frame_L_masked_straight, kernel_er, iterations=wispy_erosions)
        pixels_top = frame_L_eroded[0:int(crop_size/2)+kernel[1],:]
        pixels_bottom = frame_L_eroded[int(crop_size/2)+kernel[1]:]
        rotate_angle, face_left, depth_ratio, history_x, history_y, x_tip, y_tip, flip = \
        correct_flip('2D', frame_num - start_frame, face_left, pixels_top,pixels_bottom, history_x, history_y, cxL, cyL, ellipse, ellipse_width, \
                 width_thresh=width_thresh, speed_thresh=speed_thresh, wispy_thresh = wispy_thresh) 
        #print('DR' + str(depth_ratio))
        #print('EW '+ str(ellipse_width))
        if flip:
            print('frame ' + str(frame_num-start_frame))
        M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1)
        
        #straighten data images
        frame_L_masked_straight = cv2.warpAffine(frame_L_masked_cropped,M,square_size)
        if not video_type == '2D':
            stereo_image_straight = cv2.warpAffine(stereo_image_smoothed,M,square_size)

        # use center of mouse and orientation to get get mouse velocity relative to orientation
        delta_x = cxL - cxL_prev
        delta_y = - cyL + cyL_prev
    
        vec_length = np.sqrt(x_tip**2+y_tip**2)
        head_dir = [x_tip/vec_length,-y_tip/vec_length]
        head_dir_ortho = [-y_tip/vec_length,-x_tip/vec_length]
        vel_along_head_dir = np.dot([delta_x,delta_y],head_dir)  
        vel_ortho_head_dir = np.dot([delta_x,delta_y],head_dir_ortho)            
        #print(vel_along_head_dir)
        # ---------------------
        # Save and display data
        # ---------------------      
        
        # save data for further analysis
        if save_data:
            if save_2D_data:
                data_video_2D.write(frame_L_masked_straight)
            if video_type == '2D':
                mouse_velocity.append([vel_along_head_dir,vel_ortho_head_dir,head_dir[0],head_dir[1]])
            else:
                data_video_3D.write(stereo_image_straight)  
                mouse_velocity.append([vel_along_head_dir,vel_ortho_head_dir,head_dir[0],head_dir[1]])
            data_times = np.append(data_times,frame_num-1-time_filter_width*(not video_type=='2D')) #minus 1 to put in python coordinates
            mouse_coordinates.append([cxL,cyL])
            disruptions.append(disruption)
            disruption = 0
            

        # prepare frames for our vieweing pleasure
        if write_images or show_images:
            frame_L_masked_straight_resized = cv2.resize(frame_L_masked_straight,(crop_size*3,crop_size*3))  

            if not video_type == '2D':
                frame_R_masked_resized = cv2.resize(frame_R_masked_cropped,(crop_size*3,crop_size*3))
                stereo_image_L_resized = cv2.resize(stereo_image_L_cropped,(crop_size*3,crop_size*3)) 
                stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(crop_size*3,crop_size*3))    
                stereo_image_combined_resized = cv2.resize(stereo_image_combined,(crop_size*3,crop_size*3))
                stereo_image_combined_resized = cv2.applyColorMap(stereo_image_combined_resized, cv2.COLORMAP_OCEAN)
                cv2.ellipse(stereo_image_smoothed,ellipse,(100,255,100),1)
                cv2.circle(stereo_image_smoothed,(int(ellipse[0][0] + x_tip ),int(ellipse[0][1] + y_tip )), 
                                                                radius=3, color=(255,255,255),thickness=5) 
                stereo_image_smoothed_show = cv2.resize(stereo_image_smoothed,(crop_size*3,crop_size*3))
                stereo_image_smoothed_show = cv2.applyColorMap(stereo_image_smoothed_show, cv2.COLORMAP_OCEAN)
                stereo_image_straight = cv2.resize(stereo_image_straight,(crop_size*3,crop_size*3))
                stereo_image_straight = cv2.applyColorMap(stereo_image_straight, cv2.COLORMAP_OCEAN)
            
        # save videos
        if write_images:
            if write_normal_video:
                frameR = cv2.cvtColor(frame[:,:,r], cv2.COLOR_GRAY2RGB)
                normal_video.write(frameR)   
            if write_normalized_video:
                frame_norm_R = cv2.cvtColor(frame_norm_R, cv2.COLOR_GRAY2RGB)
                normalized_video.write(frame_norm_R)
            if write_cropped_mice:
                frame_norm_R_resized = cv2.cvtColor(frame_norm_R_resized, cv2.COLOR_GRAY2RGB)
                cropped_mouse.write(frame_R_masked_resized)
            if write_stereo_inputs:
                frame_L_masked_striped = cv2.cvtColor(frame_L_masked_striped, cv2.COLOR_GRAY2RGB)
                stereo_input_L.write(frame_L_masked_striped)
                frame_R_masked_striped = cv2.cvtColor(frame_R_masked_striped, cv2.COLOR_GRAY2RGB)
                stereo_input_R.write(frame_R_masked_striped)
            if write_3D_combined:
                threeD_combined.write(stereo_image_combined_resized)
            if write_3D_smooth: 
                threeD_smooth.write(stereo_image_smoothed)    
      
        # display videos
        if show_images:
            
#            cv2.imshow('2D', frame[:,:,l])
            cv2.imshow('2D_norm', frame_norm_L)
#            cv2.imshow('2D_norm_croppedL', frame_L_masked) 
            cv2.imshow('2D straight', frame_L_masked_straight_resized)
            
            if not video_type == '2D':
#               cv2.imshow('2D_norm_croppedR', frame_R_masked_resized)
               cv2.imshow('2D_norm_R', frame_norm_R)
#               cv2.imshow('right',frame_R_masked)
               cv2.imshow('maskcheckL',frame_L_masked_striped)
               cv2.imshow('maskcheckR',frame_R_masked_striped)
#               cv2.imshow('3D left', stereo_image_L_resized) 
#               cv2.imshow('3D right', stereo_image_R_resized) 
#               cv2.imshow('3D combined', stereo_image_combined_resized)    
#               cv2.imshow('3D combined_gauss', stereo_image_smoothed_show)
               cv2.imshow('3D straight', stereo_image_straight)

    

        # stop when 'q' is pressed
        if show_images:
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                print(vid.get(cv2.CAP_PROP_POS_FRAMES))
                break
        
        #stop at end frame
        if frame_num >= end_frame:
            break 
        
        #display notification evert 500 frames
        if (frame_num-start_frame)%500==0:
            print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
        
    else:
        print('frame-grabbing problem!!')

# ----------------------
# Wrap up and save data
# ----------------------

# save position, velocity, and frame numbers
if save_data:
    if not video_type == '2D':
        data_video_3D.release()
    if save_2D_data:
        data_video_2D.release()
    np.save(file_loc + '_coordinates.npy', mouse_coordinates)
    mouse_velocity[0][0:2] = [0,0]
    np.save(file_loc + '_velocity.npy', mouse_velocity)
    np.save(file_loc + '_frames', data_times)
    np.save(file_loc + '_disruption', disruptions)

# end videos being read and saved
vid.release()
if write_images:
    if write_normal_video:
        normal_video.release()
    if write_normalized_video:
        normalized_video.release()
    if write_cropped_mice:
        cropped_mouse.release()
    if write_stereo_inputs: 
        stereo_input_L.release()     
        stereo_input_R.release() 
    if write_3D_combined:
        threeD_combined.release()
    if write_3D_smooth:
        threeD_smooth.release()

# close all windows
#cv2.destroyAllWindows()


  