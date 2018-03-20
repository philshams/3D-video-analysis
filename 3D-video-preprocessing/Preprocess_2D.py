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

file_loc = 'C:\Drive\Video Analysis\data\\baseline_analysis\\'
date = '27.02.2018\\'
mouse_session = '205_1a\\'
save_vid_name = 'normal_1_' # name-tag to be associated with all saved files
file_name = 'Chronic_Mantis_stim-default-996386-video-'
videos = [0,1,2]

undistort = True
shelter = True
file_loc = file_loc + date + mouse_session 


# --------------------
# Set video parameters
# --------------------
frame_rate = 40 #aesthetic frame rate of saved videos
display_frame_rate = 1000 #1000 makes it as fast as possible
start_frame = 0
end_frame = np.inf #set as either desired end frame or np.inf to go to end of movie


# ----------------------------
# Set mouse contour parameters
# ----------------------------
mask_thresh = .42 #mouse mask threshold (lower is more stringently darker than background)
kernel = [4,3] #erosion and dilation kernel sizes for mouse mask
iters = [0,7] #number of erosion and dilation iterations for mouse mask


# -----------------------
# Set analysis parameters
# -----------------------
use_norm_to_analyze = True


# ------------------------------------------
# Set mouse orientation detection parameters
# ------------------------------------------
wispy_thresh = 1.2 #1.15, 1.5
wispy_erosions = 8 #12
speed_thresh = 6.5
width_thresh= 1.2 #1.2


# -----------------------
# Set data-saving options
# -----------------------
show_images = False
save_data = True
write_images = False
do_not_overwrite = False



if write_images:
    write_normal_video = False
    write_normalized_video = True
    write_cropped_mice = False




#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Inititialization -- Functions to run once per video                 -----------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -----------------------
# Perform initialization?
# -----------------------
initial_run = False

if initial_run:
    do_get_background = True #for background subtraction
    do_get_shelter = True
else:
    do_get_background = False
    do_get_shelter = False


# ----------------------------------------
# get or load background subtraction image
# ----------------------------------------
if do_get_background:
    file = file_loc + file_name + str(videos[0]) +'.avi'
    vid = cv2.VideoCapture(file)
    background_mat = get_background_mean(vid, None, False, False, file_loc = file_loc, avg_over = 100)
    vid.release()
else:
    background_file_name = file_loc + 'background_mat_avg.npy'
    background_mat = np.load(background_file_name)

# -------------------------------------------------------------
# get shelter ROI (if dark or obscured in any part of data set)
# -------------------------------------------------------------
if do_get_shelter and shelter:
    vid = cv2.VideoCapture(file)
    ret, frame = vid.read()
    if undistort:
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    shelter_roi = cv2.selectROI(frame[:,:,1])
    np.save(file_loc + 'shelter_roi.npy', shelter_roi)
    vid.release()
elif shelter:
    shelter_file_name = file_loc + 'shelter_roi.npy'
    shelter_roi = np.load(shelter_file_name)






#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Set up analysis                -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ----------------------
# Set up analysis videos
# ----------------------
crop_size = 150
square_size = (crop_size,crop_size)
border_size = 150


if undistort == True:
    maps = np.load(file_loc + 'fisheye_maps.npy')
    map1 = maps[:,:,0:2]
    map2 = maps[:,:,2]


# create background for subtraction
background_L = background_mat[:,:,0]
shelter_x_coords = [shelter_roi[0], shelter_roi[0]+shelter_roi[2]]
shelter_y_coords = [shelter_roi[1], shelter_roi[1]+shelter_roi[3]]

# initialize mouse video
for v in videos:
        
    # load video
    file = file_loc + file_name + str(v) +'.avi'
    vid = cv2.VideoCapture(file)
    print(file + ' loaded')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    end_frame = min(end_frame, vid.get(cv2.CAP_PROP_FRAME_COUNT)-5)
    if width==0:
        raise Exception('Video file not found')
    vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    out_of_bound = 0
    
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
    save_vid_name_cur = save_vid_name + str(v)
    file_loc_save = file_loc + save_vid_name_cur
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #LJPG for lossless, XVID or MJPG works for compressed
    
    # write videos as selected above
    if write_images:
        normal_video, normalized_video, cropped_mouse, stereo_input_L, stereo_input_R, threeD_combined, threeD_smooth = \
                        write_videos(file_loc_save, write_images, do_not_overwrite, fourcc, frame_rate, width, height, crop_size, border_size,
                         write_normal_video, write_normalized_video, write_cropped_mice, False, False, False)
    
    # save data for further analysis
    if save_data == True:
        fourcc = cv2.VideoWriter_fourcc(*'LJPG') #LJPG for lossless
        data_file = file_loc_save + '_data.avi'
        if os.path.isfile(data_file) and do_not_overwrite:
            raise Exception('File already exists') 
        data_video = cv2.VideoWriter(data_file,fourcc , frame_rate, (crop_size,crop_size), False) 
        data_times = np.array([])
        mouse_coordinates = []
        mouse_velocity = []
        disruptions = []
        out_of_bounds = [] #0 is in bounds, 1 is in shelter, 2 is other
     
        
    
       
              
    #%% -------------------------------------------------------------------------------------------------------------------------------------
    #------------------------                   Perform analysis                ------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    

    while True:
        # grab the frame
        ret, frame = vid.read()
        frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES) 

        if ret:  

            # ------------------------------
            # Perform background subtraction
            # -------------------------------
            # separate left and right frames
            frame_L = frame[:,:,1] 
            
            # divide image by the average background to emphasize the mouse
            frame_norm_L = (256/2 * frame_L / background_L ).astype(np.uint8)
            
            #rectify image
            if undistort == True:
                frame_norm_L = cv2.remap(frame_norm_L, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
              
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
            
            # ---------------------------------------------------------
            # If the mouse is obscured or in the shelter, skip analysis
            # ---------------------------------------------------------
            try:
                contoursL, big_cnt_indL, cxL, cyL, cntL = get_biggest_contour(frame_norm_L_mask)
            except:
                disruption = 1
                if cyL < max(shelter_y_coords + 20):
                    if out_of_bound!=1:
                        print('out of bounds, sheltered')
                    out_of_bound = 1
                else:
                    out_of_bound = 2
                out_of_bounds.append(out_of_bound)
                continue
            
            if min(shelter_x_coords) <= cxL <= max(shelter_x_coords) and min(shelter_y_coords) <= cyL <= max(shelter_y_coords):
                disruption = 1
                if out_of_bound!=1:
                    print('sheltered...')
                if show_images:
                    cv2.imshow('2D_norm', frame_norm_L)
                    if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                        break
                out_of_bound = 1
                out_of_bounds.append(out_of_bound)
                continue
            else:
                out_of_bound = 0
                
                
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
            frame_L_masked_cropped = frame_L_masked_padded[cyL-int(crop_size/2)+border_size:cyL+int(crop_size/2)+border_size,cxL-int(crop_size/2)+border_size:cxL+int(crop_size/2)+border_size]    
                
    
            # --------------------------
            # Get speed and orientation
            # --------------------------
            # flip mouse into the correct orientation
            try:
                rotate_angle, face_left, ellipse, topright_or_botleft, ellipse_width = \
                flip_mouse(face_left, ellipse, topright_or_botleft,frame_L_masked_cropped , sausage_thresh = 1.1)
            except:
                raise Exception('mouse out of bounds but not captured above...')
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
            
            cv2.line(frame_L_eroded,(0, 75+kernel[1]),(150, 75+kernel[1]),color=250)
          
            #print('Dep Rat ' + str(depth_ratio))
            #print('Ell Wid '+ str(ellipse_width))
            if flip:
                print('frame ' + str(frame_num-start_frame))
            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1)
            
            #straighten data images
            frame_L_masked_straight = cv2.warpAffine(frame_L_masked_cropped,M,square_size)

    
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
                data_video.write(frame_L_masked_straight)
                mouse_velocity.append([vel_along_head_dir,vel_ortho_head_dir,head_dir[0],head_dir[1]])
                data_times = np.append(data_times,frame_num-1) #minus 1 to put in python coordinates
                mouse_coordinates.append([cxL,cyL])
                disruptions.append(disruption)
                out_of_bounds.append(out_of_bound)
                disruption = 0
                
                
            # prepare frames for our vieweing pleasure
            if write_images or show_images:
                frame_L_masked_straight_resized = cv2.resize(frame_L_masked_straight,(crop_size*3,crop_size*3))  
                
            # save videos
            if write_images:
                if write_normal_video:
                    frameL = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
                    normal_video.write(frameL)   
                if write_normalized_video:
                    frame_norm_L = cv2.cvtColor(frame_norm_L, cv2.COLOR_GRAY2RGB)
                    normalized_video.write(frame_norm_L)
                if write_cropped_mice:
                    frame_L_masked_cropped = cv2.cvtColor(frame_L_masked_cropped, cv2.COLOR_GRAY2RGB)
                    cropped_mouse.write(frame_L_masked_cropped)
     
          
            # display videos
            if show_images:
                
    #            cv2.imshow('2D', frame[:,:,l])
                cv2.imshow('2D_norm', frame_norm_L)
    #            cv2.imshow('2D_norm_croppedL', frame_L_masked_cropped) 
                cv2.imshow('2D straight', frame_L_masked_straight_resized)
                
    
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
        data_video.release()
        np.save(file_loc_save + '_coordinates.npy', mouse_coordinates)
        try:
            mouse_velocity[0][0:2] = [0,0]
        except:
            print('session spent in shelter')
        np.save(file_loc_save + '_velocity.npy', mouse_velocity)
        np.save(file_loc_save + '_frames', data_times)
        np.save(file_loc_save + '_disruption', disruptions)
        np.save(file_loc_save + '_out_of_bounds', out_of_bounds)
    
    # end videos being read and saved
    vid.release()
    if write_images:
        if write_normal_video:
            normal_video.release()
        if write_normalized_video:
            normalized_video.release()
        if write_cropped_mice:
            cropped_mouse.release()
    
    
    # close all windows
    #cv2.destroyAllWindows()


  