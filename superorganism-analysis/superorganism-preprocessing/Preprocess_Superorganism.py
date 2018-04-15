'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                Display and Analyze a saved stereoscopic video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import cv2; import os
from Depth_funcs import create_global_matchers, get_background_mean, make_striped_background, get_y_offset, get_biggest_contour, get_second_biggest_contour
from Depth_funcs import flip_mouse, correct_flip, write_videos
'''
cv2.destroyAllWindows()
'''

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------               Select video file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
video_file_location = "C:\\Drive\\Video Analysis\\data\\3D_pipeline\\bj\\"
video_name = "bj148b_test2.avi"

save_folder_location = "C:\\Drive\\Video Analysis\\data\\3D_pipeline\\"
session_name_tag = "session1"  # name-tag to be associated with all saved files



# load video ...
vid = cv2.VideoCapture(video_file_location + video_name)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  


# --------------------
# Set video parameters
# --------------------
frame_rate = 40 #aesthetic frame rate of saved videos
display_frame_rate = 1000 #1000 makes it as fast as possible
start_frame = 302
#end_frame = np.inf #set as either desired end frame or np.inf to go to end of movie
end_frame = start_frame + 12000  #10 min = 24000 frames, 3 min = 7200, 1 min = 2400
#gain = 1

# ----------------------------
# Set mouse contour parameters
# ----------------------------
mask_thresh = .44 #mouse mask threshold (lower is more stringently darker than background)
kernel = [3,4] #erosion and dilation kernel sizes for mouse mask
iters = [3,4] #number of erosion and dilation iterations for mouse mask
single_mouse_thresh = 2000 #2nd biggest contour must be at least this big (about 50% of mouse size)
double_mouse_thresh = 8000 #biggest contour must be under this size  (about 150% of mouse size)

# -----------------------------
# Set depth analysis parameters
# -----------------------------
smooth_factor = 6
final_smoothing_kernel_width = 25

window_size = 1
min_disparity = 64
num_disparities = 1*16  
max_pixel_diff = 8

pre_filter_cap = 61 
unique_ratio = 10


# ------------------------------------------
# Set mouse orientation detection parameters
# ------------------------------------------
speed_thresh = 12
depth_percentile = 60 # 40 may also work
width_thresh= 1.2
depth_ratio_thresh = [.75, .68, .6] 
pixel_value_thresh = [109,89,69]


# -----------------------
# Set data-saving options
# -----------------------
show_images = True
save_data = True #must update for super-organism
write_images = True #must update for super-organism
do_not_overwrite = False

if write_images:
    write_normal_video = False
    write_normalized_video = True
    write_cropped_mice = False
    write_stereo_inputs = False
    write_3D_combined = False
    write_3D_smooth = False




#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Inititialization -# Functions to run once per video                 -----------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -----------------------
# Perform initialization?
# -----------------------
    
initial_run = False

save_folder_location_session = save_folder_location + session_name_tag + '\\'
save_file_location_session = save_folder_location + session_name_tag + '\\' + session_name_tag

if initial_run:
    do_get_background = True
    do_get_y_offset = True
    do_get_x_offset = True
    if not os.path.isdir(save_folder_location + session_name_tag):
        os.makedirs(save_folder_location + session_name_tag)
        print("saving to " + save_folder_location + session_name_tag)   
    
    
else:
    do_get_background = False
    do_get_y_offset = False
    do_get_x_offset = False


# ----------------------------------------
# get or load background subtraction image
# ----------------------------------------

if do_get_background:
    background_mat = get_background_mean(vid, start_frame = start_frame, file_loc = save_file_location_session, avg_over = 100)
else:
    background_file_name = save_file_location_session + '_background_mat_avg.npy'
    background_mat = np.load(background_file_name)
#vid.release()



# -------------------------------------------------------------------
# get or load y-offset of cameras (cheap imitation of an calibration)
# -------------------------------------------------------------------
    
if do_get_y_offset:
    y_offset = get_y_offset(vid,background_mat,start_frame, stop_frame = 1000, l=1, r=2, mask_thresh = mask_thresh, kernel = kernel, iters = iters) #kernel and iters for erosion and dilation, respectively
    np.save(save_file_location_session + '_y_offset.npy',y_offset)
else:
    y_offset_file_name = save_file_location_session + '_y_offset.npy'
    y_offset = int(np.load(y_offset_file_name))


# ---------------------------------------------
# get or load x-offset of cameras (to come....)
# ----------------------------------------------
    
#get x-offset of mouse
#make some function a la:
#        for shift in range(100):
#            cv2.imshow('mask_overlay',frame_norm_L_masked + frame_norm_R_padded[100:-100,(100-shift):-(100+shift)])
#            if cv2.waitKey(70) & 0xFF == ord('q'):
#                break



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
crop_size_single = 250
crop_size_together = 250
border_size = 200
l = 1   #left cam is input 1, right cam is input 2...for now
r = 2


# make striped background # in size of ROI for 3D mouse detection -# height: crop_size+border_size and length: crop_size+2*border_size
stripesL_single, stripesR_single = make_striped_background(
        height, width, min_disparity, roi_height=crop_size_single + int(border_size), roi_width= int(2 * crop_size_single)) # + 2*border_size))
stripesL_together, stripesR_together = make_striped_background(
        height, width, min_disparity, roi_height=crop_size_together + int(border_size), roi_width=int(2 * crop_size_together))# + 2*border_size))

# initialize mouse video
vid = cv2.VideoCapture(video_file_location + video_name)
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
if num_frames < 1:
    print(video_file_location + video_name)
    raise Exception('Video not opening')

end_frame = min(end_frame, num_frames-5)


# create background for subtraction
background_L = background_mat[:,:,0]         
background_R = cv2.copyMakeBorder(background_mat[:,:,1],top=y_offset,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
background_R = background_R[:-y_offset,:] 

# initialize mouse orientation detection
face_left = 1
topright_or_botleft = 1
topright_or_botleft_prev = 1
depth_ratio = np.ones(3)
cxL = 400
cyL = 240
move_prev = 0
history_x = np.zeros(4)
history_y = np.zeros(4)
ellipse = 0
ellipse_width = 0
slope_recipr = 1

#initialize erosion/dilation kernels
kernel_er = np.ones((kernel[0],kernel[0]),np.uint8)
kernel_dil = np.ones((kernel[1],kernel[1]),np.uint8)
kernel_head = np.ones((9,9),np.uint8)


# ----------------------
# Select videos to save 
# ----------------------



fourcc = cv2.VideoWriter_fourcc(*'XVID') #LJPG for lossless, XVID or MJPG works for compressed

# write videos as selected above
if write_images:
    normal_video, normalized_video, cropped_mouse, stereo_input_L, stereo_input_R, threeD_combined, threeD_smooth = \
                    write_videos(save_file_location_session, write_images, do_not_overwrite, fourcc, frame_rate, width, height, crop_size_together, border_size,
                     write_normal_video, write_normalized_video, write_cropped_mice, write_stereo_inputs, write_3D_combined, write_3D_smooth)

# save data for further analysis
if save_data == True:
    fourcc = cv2.VideoWriter_fourcc(*'LJPG') #LJPG for lossless
    data_file = save_file_location_session + '_data.avi'; data_file_2D = save_file_location_session + '_data2D.avi'
    if os.path.isfile(data_file) and do_not_overwrite:
        raise Exception('File already exists') 
    data_video = cv2.VideoWriter(data_file,fourcc , frame_rate, (crop_size_together,crop_size_together), False) 
    data_video_2D = cv2.VideoWriter(data_file_2D, fourcc , frame_rate, (crop_size_together,crop_size_together), False) 
    data_times = np.array([])
    mouse_coordinates = []
    mouse_velocity = []
togetherness = []
 
    

   
          
#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Perform analysis                ------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#try:

while True:
    # grab the frame
    ret, frame = vid.read()
    frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES) 

    if ret:  
        
        # ------------------------------
        # Perform background subtractionq
        # -------------------------------
        
        # separate left and right frames
        frame_L = frame[:,:,l]
        frame_R = frame[:,:,r]
        
        # shift the right frame by the y-offset, whose value was calibrated above
        frame_R = cv2.copyMakeBorder(frame_R,top=y_offset,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
        frame_R = frame_R[:-y_offset,:]
        
        # divide image by the average background to emphasize the mouse
        frame_norm_L = (256/2 * frame_L / background_L ).astype(np.uint8)
        frame_norm_R = (256/2 * frame_R / background_R ).astype(np.uint8)

           
        # -------------------------------------
        # Find the contour around the mouse(s)
        # -------------------------------------
        
        # use the thresholds, erosion, and dilation set above to extract a mask coinciding with the mouse
        frame_norm_L_mask = ((frame_norm_L / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_R_mask = ((frame_norm_R / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=iters[0]) 
        frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=iters[0])
        frame_norm_L_mask = cv2.dilate(frame_norm_L_mask, kernel_dil, iterations=iters[1]) 
        frame_norm_R_mask = cv2.dilate(frame_norm_R_mask, kernel_dil, iterations=iters[1])

        # extract the largest contour in this mask -- should correspond to the mouse
        cxL_prev = cxL #first, save the previous center of mouse in order to extract velocity
        cyL_prev = cyL #first, save the previous center of mouse in order to extract velocity
        contoursL, big_cnt_indL, cxL, cyL, cntL = get_biggest_contour(frame_norm_L_mask)
        contoursR, big_cnt_indR, cxR, cyR, cntR = get_biggest_contour(frame_norm_R_mask)
        
        # extract the largest contour in this mask -- should correspond to the second mouse, if applicable
        scd_big_cnt_indL, cxL2, cyL2, cntL2, togetherL = get_second_biggest_contour(frame_norm_L_mask,single_mouse_thresh, double_mouse_thresh)
        scd_big_cnt_indR, cxR2, cyR2, cntR2, togetherR = get_second_biggest_contour(frame_norm_R_mask,single_mouse_thresh, double_mouse_thresh)
                
        # create a new mask, corresponding to only the largest contour
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_R = (cv2.drawContours(blank, contoursR, big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_L = (cv2.drawContours(blank, contoursL, big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED))
        
        # create a new mask, corresponding to only the 2nd largest contour
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_R2 = (cv2.drawContours(blank, contoursR, scd_big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_L2 = (cv2.drawContours(blank, contoursL, scd_big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED))
        
        # apply this mask to the original image
        frame_L_masked = frame_L * contour_mask_L
        frame_R_masked = frame_R * contour_mask_R
        frame_L_masked2 = frame_L * contour_mask_L2
        frame_R_masked2 = frame_R * contour_mask_R2
        
        
        # --------------------------------------------------------------------
        # Set settings according to whether the mice are together or separate
        # --------------------------------------------------------------------
        if togetherR or togetherL:
            crop_size = crop_size_together
            square_size = (crop_size,crop_size)
            stripesL, stripesR = stripesL_together, stripesR_together
        else:
            crop_size = crop_size_single
            square_size = (crop_size,crop_size)
            stripesL, stripesR = stripesL_single, stripesR_single
            
            
        if (togetherR or togetherL) and not (togetherR and togetherL):
            print('cameras don''t agree')
            togetherR, togetherL = False, False
            togetherness.append(int(togetherR))
            continue
        elif (togetherR and togetherL):
            print('together')
        else:
            print('apart')
            if show_images:
                cv2.imshow('2D_norm', frame_R)
                if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                    break
            togetherness.append(int(togetherR))
            continue
        
        
        # -----------------
        # Loop across mice
        # -----------------
        for mouse in [0,1]:
            if mouse==0: # and not togetherR:
                if abs(cyR-cyL) > 10 or (cxL-cxR) < 68 or (cxL-cxR) > 88:
#                    scd_big_cnt_indL, cxL2, cyL2, cntL2, big_cnt_indL, cxL, cyL, cntL = big_cnt_indL, cxL, cyL, cntL, scd_big_cnt_indL, cxL2, cyL2, cntL2
#                    contour_mask_L, contour_mask_L2,frame_L_masked, frame_L_masked2 = contour_mask_L2, contour_mask_L, frame_L_masked2, frame_L_masked
                    print(' cameras disagree ! ')
                    togetherness.append(int(0))
                    break
                    
            if mouse==1:
                if togetherR: #don't plot or analyze the second mouse if they're together
                    continue
#                else:
#                    scd_big_cnt_indL, cxL2, cyL2, cntL2, big_cnt_indL, cxL, cyL, cntL = big_cnt_indL, cxL, cyL, cntL, scd_big_cnt_indL, cxL2, cyL2, cntL2
#                    contour_mask_L, contour_mask_L2,frame_L_masked, frame_masked2 = contour_mask_L2, contour_mask_L, frame_L_masked2, frame_L_masked
#                    scd_big_cnt_indR, cxR2, cyR2, cntR2, big_cnt_indR, cxR, cyR, cntR = big_cnt_indR, cxR, cyR, cntR, scd_big_cnt_indR, cxR2, cyR2, cntR2
#                    contour_mask_R, contour_mask_R2,frame_R_masked, frame_R_masked2 = contour_mask_R2, contour_mask_R, frame_R_masked2, frame_R_masked
            
    
            # create a striped mask of size set above, in which to place the image of the mouse that was just extracted 
            contour_mask_L_padded = cv2.copyMakeBorder(contour_mask_L,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            contour_mask_R_padded = cv2.copyMakeBorder(contour_mask_R,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            reduced_contour_mask_L = contour_mask_L_padded[cyL+int(border_size/2-crop_size/2):cyL+int(3*border_size/2 + crop_size/2),cxL+border_size-crop_size:cxL+crop_size + border_size]
            reduced_contour_mask_R = contour_mask_R_padded[cyL+int(border_size/2-crop_size/2):cyL+int(3*border_size/2 + crop_size/2),cxL+border_size-crop_size:cxL+crop_size + border_size]
            stripe_maskL = stripesL * (1 - reduced_contour_mask_L)
            stripe_maskR = stripesR * (1 - reduced_contour_mask_R)
            
            # lay the mouse on top of this striped background
            frame_L_masked_padded = cv2.copyMakeBorder(frame_L_masked,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            frame_R_masked_padded = cv2.copyMakeBorder(frame_R_masked,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            frame_L_masked_striped = frame_L_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxL+border_size-crop_size:cxL+crop_size + border_size] + stripe_maskL
            frame_R_masked_striped = frame_R_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxL+border_size-crop_size:cxL+crop_size + border_size] + stripe_maskR
            

            # --------------------------
            # Compute 3D image of mouse
            # --------------------------    
            
            # compute stereo images from each direction
            stereo_image_L = stereo_left.compute(frame_L_masked_striped,frame_R_masked_striped).astype(np.uint8)
            stereo_image_R = stereo_right.compute(frame_R_masked_striped,frame_L_masked_striped).astype(np.uint8)
            
            # crop the stereo images to the crop size set above
            stereo_image_L_cropped = stereo_image_L[int(border_size/2):int(border_size/2+crop_size), int(crop_size/2):int(3*crop_size/2)]
            stereo_image_R_cropped = stereo_image_R[int(border_size/2):int(border_size/2+crop_size), int(crop_size/2)-(cxL-cxR):int(3*crop_size/2)-(cxL-cxR)]
            
            # crop the original, masked image to the same size
            frame_R_masked_cropped = frame_R_masked_padded[cyR-int(crop_size/2)+border_size:cyR+int(crop_size/2)+border_size,cxR-int(crop_size/2)+border_size:cxR+int(crop_size/2)+border_size]
            frame_L_masked_cropped = frame_L_masked_padded[cyL-int(crop_size/2)+border_size:cyL+int(crop_size/2)+border_size,cxL-int(crop_size/2)+border_size:cxL+int(crop_size/2)+border_size]    
            
            # combine the two stereo images into an average, on the pixels on which both cameras agree there was a mouse
            stereo_image_combined = ((stereo_image_L_cropped + (255 - stereo_image_R_cropped))*(frame_R_masked_cropped>0)*(frame_L_masked_cropped>0)).astype(np.uint8)
    
    
            # ------------------------
            # Smooth 3D image of mouse
            # ------------------------
            # do Gaussian smoothing on the current, combined stereo image
            stereo_image_smoothed = cv2.GaussianBlur(stereo_image_combined,ksize=(final_smoothing_kernel_width,final_smoothing_kernel_width),
                                                                   sigmaX=final_smoothing_kernel_width,sigmaY=final_smoothing_kernel_width)
            

            # ----------------------
            # Get mouse orientation
            # ----------------------
            # flip mouse into the correct orientation
            rotate_angle, face_left, ellipse, topright_or_botleft, ellipse_width = \
            flip_mouse(face_left, ellipse, topright_or_botleft, stereo_image_smoothed, sausage_thresh = 1.1)
            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1) 
            stereo_image_straight = cv2.warpAffine(stereo_image_smoothed,M,(crop_size,crop_size))        
    
            # check for errors -- if the tail end is more elevated, or the mouse is running toward its tail, slip 180 degrees
            stereo_top = stereo_image_straight[0:int(crop_size/2),:]
            stereo_bottom = stereo_image_straight[int(crop_size/2):]
            rotate_angle, face_left, depth_ratio, history_x, history_y, x_tip, y_tip, flip = \
            correct_flip(frame_num - start_frame, face_left, stereo_top,stereo_bottom, depth_percentile, depth_ratio, history_x, history_y, cxL, cyL, ellipse, ellipse_width, \
                     width_thresh=width_thresh, speed_thresh=speed_thresh, depth_ratio_thresh = depth_ratio_thresh, pixel_value_thresh = pixel_value_thresh)   
            if flip:
                print('frame ' + str(frame_num-start_frame))
            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1)
            stereo_image_straight = cv2.warpAffine(stereo_image_smoothed,M,square_size)  
            frame_L_masked_cropped_straight = cv2.warpAffine(frame_L_masked_cropped,M,square_size) 
            
            # use center of mouse and orientation to get get mouse velocity relative to orientation
            delta_x = cxL - cxL_prev
            delta_y = - cyL + cyL_prev
        
            vec_length = np.sqrt(x_tip**2+y_tip**2)
            head_dir = [x_tip/vec_length,-y_tip/vec_length]
            head_dir_ortho = [-y_tip/vec_length,-x_tip/vec_length]
            vel_along_head_dir = np.dot([delta_x,delta_y],head_dir)  
            vel_ortho_head_dir = np.dot([delta_x,delta_y],head_dir_ortho)
            
            
            # ---------------------
            # Save and display data
            # --------------------#      
            
            # save data for further analysis
            if save_data and togetherR:
                data_video.write(stereo_image_straight)  
                data_video_2D.write(frame_L_masked_cropped_straight) 
                data_times = np.append(data_times,frame_num-1) #minus 1 to put in python coordinates
                mouse_coordinates.append([cxL,cyL])
                mouse_velocity.append([vel_along_head_dir,vel_ortho_head_dir,head_dir[0],head_dir[1]])
            togetherness.append(int(togetherR))
    
            # prepare frames for our vieweing pleasure
            if write_images or show_images:
                frame_norm_R_resized = cv2.resize(frame_R_masked_cropped,(crop_size*3,crop_size*3))
                frame_norm_L_resized = cv2.resize(frame_L_masked_cropped,(crop_size*3,crop_size*3))  
    
                stereo_image_L_resized = cv2.resize(stereo_image_L_cropped,(crop_size*3,crop_size*3)) 
                stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(crop_size*3,crop_size*3))    
                stereo_image_combined_resized = cv2.resize(stereo_image_combined,(crop_size*3,crop_size*3))
                stereo_image_combined_resized = cv2.applyColorMap(stereo_image_combined_resized, cv2.COLORMAP_OCEAN)
                cv2.ellipse(stereo_image_smoothed,ellipse,(100,255,100),1)
                cv2.circle(stereo_image_smoothed,(int(ellipse[0][0] + x_tip ),
                                                                int(ellipse[0][1] + y_tip )), 
                                                                radius=3, color=(255,255,255),thickness=5) 
                stereo_image_smoothed_show = cv2.resize(stereo_image_smoothed,(crop_size*3,crop_size*3))
                stereo_image_smoothed_show = cv2.applyColorMap(stereo_image_smoothed_show, cv2.COLORMAP_OCEAN)
                stereo_image_straight = cv2.resize(stereo_image_straight,(crop_size*3,crop_size*3))
                stereo_image_straight = cv2.applyColorMap(stereo_image_straight, cv2.COLORMAP_OCEAN)
                
            # save videos
            if write_images and togetherR:
                
                if write_normal_video:
                    frameR = cv2.cvtColor(frame[:,:,r], cv2.COLOR_GRAY2RGB)
                    normal_video.write(frameR)   
                if write_normalized_video:
                    frame_R = cv2.cvtColor(frame_R, cv2.COLOR_GRAY2RGB)
                    normalized_video.write(frame_R)
                if write_cropped_mice:
                    frame_norm_R_resized = cv2.cvtColor(frame_norm_R_resized, cv2.COLOR_GRAY2RGB)
                    cropped_mouse.write(frame_norm_R_resized)
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
                
#                cv2.imshow('maskcheckL',frame_L_masked_striped)
#                cv2.imshow('maskcheckR',frame_R_masked_striped)
    #            cv2.imshow('2D', frame[:,:,r])
                cv2.imshow('2D_norm', frame_R)
                cv2.imshow('2D_norm_croppedR', frame_norm_R_resized)
    #            cv2.imshow('2D_norm_croppedL', frame_norm_L_resized) 
    #            cv2.imshow('3D left', stereo_image_L_resized) 
    #            cv2.imshow('3D right', stereo_image_R_resized) 
    #            cv2.imshow('3D combined', stereo_image_combined_resized)    
    #            cv2.imshow('3D combined_gauss', stereo_image_smoothed_show)
                cv2.imshow('3D straight', stereo_image_straight)

        # stop when 'q' is pressed
        if show_images:
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
        
        #stop at end frame
        if frame_num >= end_frame:
            break 
        
        #display notification evert 500 frames
        if (frame_num-start_frame)%500==0:
            print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
        
    else:
        print('frame-grabbing problem!!')
        cv2.waitKey(1000)
#except:
#    cv2.waitKey()
#    cv2.destroyAllWindows()

# ----------------------
# Wrap up and save data
# ----------------------
print(vid.get(cv2.CAP_PROP_POS_FRAMES))
# save position, velocity, and frame numbers
if save_data:
    data_video.release()
    data_video_2D.release()
    if os.path.isfile(save_file_location_session + '_frames') and not do_not_overwrite:
        raise Exception('File already exists')
    np.save(save_file_location_session + '_coordinates.npy', mouse_coordinates)
    np.save(save_file_location_session + '_velocity.npy', mouse_velocity)
    np.save(save_file_location_session + '_frames', data_times)
    np.save(save_file_location_session + '_together', togetherness)
    np.save(save_file_location_session + '_start_end_frame', [start_frame, frame_num])

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
cv2.waitKey()
cv2.destroyAllWindows()

 

  