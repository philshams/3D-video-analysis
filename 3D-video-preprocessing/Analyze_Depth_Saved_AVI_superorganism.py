#%load_ext autoreload
#%autoreload 2

#%%Stereoscopic AVI file to be analyzed
import numpy as np
import cv2
import os
from Depth_funcs import create_global_matchers, get_background_mean, make_striped_background, get_y_offset, get_biggest_contour, get_second_biggest_contour
from Depth_funcs import flip_mouse, correct_flip

# location of stereoscopic video data
file_name = '3Dtest_secondmouse0.avi'
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '14.02.2018_zina\\'
mouse_session = 'twomouse\\' 

#file_name = 'mouse0.avi'
#file_loc = 'C:\Drive\Video Analysis\data\\'
#date = '05.02.2018\\'
#mouse_session = '202-1a\\'

save_vid_name = 'test'

file_loc = file_loc + date + mouse_session
file_name = file_loc + file_name
print(file_name)

# get image sizes
vid = cv2.VideoCapture(file_name)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  
frame_rate = 20

# analysis parameters
display_frame_rate = 100 #1000 makes it as fast as possible
start_frame = 1000
end_frame = min(np.inf, vid.get(cv2.CAP_PROP_FRAME_COUNT)-10) #replace np.inf with end frame  

l = 1   #left cam is input 1, right cam is input 2...for now
r = 2

#mouse masking
mask_thresh = .44 #mouse mask threshold (lower is more stringently darker than background)
kernel = [3,5] #erosion and dilation kernel sizes for mouse mask
iters = [2,4] #erosion and dilation iterations for mouse mask

#depth analysis
window_size = 1
min_disparity = 64
num_disparities = 1*16  
smooth_factor = 8
pre_filter_cap = 61 
unique_ratio = 10
max_pixel_diff = 8
final_smoothing_kernel_width = 17

#directionality analysis
depth_percentile = 60 #40?
width_thresh= 1.2
speed_thresh= np.inf
depth_ratio_thresh = [.75, .68, .6] #.8, mean?
pixel_value_thresh = [109,89,69]

#data options
show_images = True

save_data = False

write_images = False

do_not_overwrite = True

write_normal_video = True
write_normalized_video = False
write_cropped_mice = True
write_stereo_inputs = False
write_3D_combined = False
write_3D_smooth = True
write_3D_straight = True

# Get one-time parameters?
initial_run = False
if initial_run:
    do_get_background = True
    do_get_y_offset = True
    do_get_x_offset = True
else:
    do_get_background = False
    do_get_y_offset = False
    do_get_x_offset = False



#%% do initializations

# get or load 'background_mat_avg.npy' in data folder, for background subtraction
if do_get_background:
    background_mat = get_background_mean(file_name = file_name, file_loc = file_loc, avg_over_approx = 10000)
else:
    background_file_name = file_loc + 'background_mat_avg.npy'
    background_mat = np.load(background_file_name)

# get y-offset of cameras
if do_get_background:
    y_offset = get_y_offset(file_name,background_mat,start_frame, stop_frame = 1000, l=1, r=2, mask_thresh = mask_thresh, kernel = kernel, iters = iters) #kernel and iters for erosion and dilation, respectively
    np.save(file_loc + 'y_offset.npy',y_offset)
else:
    y_offset_file_name = file_loc + 'y_offset.npy'
    y_offset = int(np.load(y_offset_file_name))


#get x-offset of mouse
#make some function a la:
#        for shift in range(100):
#            cv2.imshow('mask_overlay',frame_norm_L_masked + frame_norm_R_padded[100:-100,(100-shift):-(100+shift)])
#            if cv2.waitKey(70) & 0xFF == ord('q'):
#                break


# create 3D image matcher
stereo_left, stereo_right, min_disparity, shift_pixels = create_global_matchers(window_size = window_size, min_disparity =min_disparity, 
                                                                                num_disparities = num_disparities, smooth_factor = smooth_factor,
                                                                                 pre_filter_cap = pre_filter_cap, unique_ratio = unique_ratio,
                                                                                 max_pixel_diff = max_pixel_diff)


# select videos to save 
fourcc = cv2.VideoWriter_fourcc(*'XVID') #LJPG for lossless, MJP2 /MJPG works; try MJP2 or LAGS or 'Y16 '; want uncompressed!!
file_loc = file_loc + save_vid_name

if write_images == True:
    if write_normal_video == True:
        video_file = file_loc + 'normal_video.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists') 
        normal_video = cv2.VideoWriter(video_file,fourcc , frame_rate, (width,height))#, False) 
    if write_normalized_video == True:
        video_file = file_loc + 'normalized_video.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists')         
        normalized_video = cv2.VideoWriter(video_file,fourcc , frame_rate, (width,height))#, False) 
    if write_cropped_mice == True:
        video_file = file_loc + 'cropped_mouse.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists')         
        cropped_mouse = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450))#, False) 
    if write_stereo_inputs == True:
        video_fileL = file_loc + 'stereo_input_L.avi'
        video_fileR = file_loc + 'stereo_input_R.avi'
        if (os.path.isfile(video_fileL) or os.path.isfile(video_fileR)) and do_not_overwrite:
            raise Exception('File already exists') 
        stereo_input_L = cv2.VideoWriter(video_fileL,fourcc , frame_rate, (350,200))#, False) 
        stereo_input_R = cv2.VideoWriter(video_fileR,fourcc , frame_rate, (350,200))#, False) 
    if write_3D_combined == True:
        video_file = file_loc + '3D_combined.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists')         
        threeD_combined = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450))#, True) 
    if write_3D_smooth == True:
        video_file = file_loc + '3D_smooth.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists') 
        threeD_smooth = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450))#, True)
    if write_3D_straight == True:
        video_file = file_loc + '3D_straight.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists') 
        threeD_straight = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450))#, True)
        
if save_data == True:
    fourcc = cv2.VideoWriter_fourcc(*'LJPG') #LJPG for lossless, MJP2 /MJPG works; try MJP2 or LAGS or 'Y16 '; want uncompressed!!
    data_file = file_loc + '_data.avi'
    if os.path.isfile(data_file) and do_not_overwrite:
        raise Exception('File already exists') 
    data_video = cv2.VideoWriter(data_file,fourcc , frame_rate, (150,150), False) 
    
    data_times = np.array([])
    mouse_coordinates = []
 




#cv2.destroyAllWindows()
    

#%% Run 3D analyser over frames
    
#could be anything
slope_recipr = 1

#cropped image size
crop_size_single = 150
crop_size_together = 250
border_size = 200

# make striped background - in size of ROI for 3D mouse detection
stripesL_single, stripesR_single = make_striped_background(
        height, width, min_disparity, roi_height=crop_size_single + int(border_size), roi_width= int(crop_size_single + 2*border_size))
stripesL_together, stripesR_together = make_striped_background(
        height, width, min_disparity, roi_height=crop_size_together + int(border_size), roi_width=int(crop_size_together + 2*border_size))

#set up background image
background_L = background_mat[:,:,0]          
background_R = cv2.copyMakeBorder(background_mat[:,:,1],top=y_offset,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
background_R = background_R[:-y_offset,:]    
          
#initialize mouse video
vid = cv2.VideoCapture(file_name)
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

    
#initialize head-spotting
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

#initialize erosion/dilation kernels
kernel_er = np.ones((kernel[0],kernel[0]),np.uint8)
kernel_dil = np.ones((kernel[1],kernel[1]),np.uint8)
kernel_head = np.ones((9,9),np.uint8)
        
#save test mouse depth vid as array (150 x 150 x 50)
depth_array = np.zeros((20,20,50))
fra = 0

#initialize separate vs together discrimination
come_together = 0

while True:
    ret, frame = vid.read() # get the frame
    frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES) 

    if ret:       
        frame_L = frame[:,:,l] #pull the left and right frames
        frame_R = frame[:,:,r]
        
        #y-offset, as calibrated above
        frame_R = cv2.copyMakeBorder(frame_R,top=y_offset,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
        frame_R = frame_R[:-y_offset,:]
        
        #background division to extract the mouse
        frame_norm_L = (256/2 * frame_L / background_L ).astype(np.uint8)
        frame_norm_R = (256/2 * frame_R / background_R ).astype(np.uint8)
                  
        #get the mouse mask
        frame_norm_L_mask = ((frame_norm_L / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_R_mask = ((frame_norm_R / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=iters[0]) 
        frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=iters[0])
        frame_norm_L_mask = cv2.dilate(frame_norm_L_mask, kernel_dil, iterations=iters[1]) 
        frame_norm_R_mask = cv2.dilate(frame_norm_R_mask, kernel_dil, iterations=iters[1])


        #get biggest contour from the left and right masked images
        single_mouse_thresh = 1000 #2nd biggest contour must be at least this big
        double_mouse_thresh = 7000 #biggest contour must be under this size
        contoursL, big_cnt_indL, cxL, cyL, cntL = get_biggest_contour(frame_norm_L_mask)
        contoursR, big_cnt_indR, cxR, cyR, cntR = get_biggest_contour(frame_norm_R_mask)
        scd_big_cnt_indL, cxL2, cyL2, cntL2, togetherL = get_second_biggest_contour(frame_norm_L_mask,single_mouse_thresh, double_mouse_thresh)
        scd_big_cnt_indR, cxR2, cyR2, cntR2, togetherR = get_second_biggest_contour(frame_norm_R_mask,single_mouse_thresh, double_mouse_thresh)
        
        #get and apply mask for the biggest contour
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_R = (cv2.drawContours(blank, contoursR, big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_R_masked = frame_norm_R * contour_mask_R
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_L = (cv2.drawContours(blank, contoursL, big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_L_masked = frame_norm_L * contour_mask_L

        #get and apply mask for the 2nd biggest contour
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_R2 = (cv2.drawContours(blank, contoursR, scd_big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_R_masked2 = frame_R * contour_mask_R2
        
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_L2 = (cv2.drawContours(blank, contoursL, scd_big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_L_masked2 = frame_L * contour_mask_L2
        
        #frame_norm_R_masked2 = frame_norm_R * contour_mask_R2
        #frame_norm_L_masked2 = frame_norm_L * contour_mask_L2
        cv2.imshow('testR',frame_norm_R*contour_mask_R)
        cv2.imshow('testL',frame_norm_L*contour_mask_L)
        cv2.imshow('testR2',frame_norm_R*contour_mask_R2)
        cv2.imshow('testL2',frame_norm_L*contour_mask_L2)
        
        print(togetherR)
        print(np.sum(contour_mask_R))
        print(np.sum(contour_mask_R2))
        print('')
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
            continue
        
        for mouse in [0,1]:
            if mouse==0 and not togetherR:
                if abs(cyR-cyL) > 10 or (cxL-cxR) < 68 or (cxL-cxR) > 88:
                    scd_big_cnt_indL, cxL2, cyL2, cntL2, big_cnt_indL, cxL, cyL, cntL = big_cnt_indL, cxL, cyL, cntL, scd_big_cnt_indL, cxL2, cyL2, cntL2
                    contour_mask_L, contour_mask_L2,frame_norm_L_masked, frame_norm_L_masked2 = contour_mask_L2, contour_mask_L, frame_norm_L_masked2, frame_norm_L_masked
                    print(' mouse swap! ')
            if mouse==1:
                if togetherR:
                    continue
                else:
                    scd_big_cnt_indL, cxL2, cyL2, cntL2, big_cnt_indL, cxL, cyL, cntL = big_cnt_indL, cxL, cyL, cntL, scd_big_cnt_indL, cxL2, cyL2, cntL2
                    contour_mask_L, contour_mask_L2,frame_norm_L_masked, frame_norm_L_masked2 = contour_mask_L2, contour_mask_L, frame_norm_L_masked2, frame_norm_L_masked
                    scd_big_cnt_indR, cxR2, cyR2, cntR2, big_cnt_indR, cxR, cyR, cntR = big_cnt_indR, cxR, cyR, cntR, scd_big_cnt_indR, cxR2, cyR2, cntR2
                    contour_mask_R, contour_mask_R2,frame_norm_R_masked, frame_norm_R_masked2 = contour_mask_R2, contour_mask_R, frame_norm_R_masked2, frame_norm_R_masked
            

            #pad the contour map, get it to the right size
            contour_mask_L_padded = cv2.copyMakeBorder(contour_mask_L,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            contour_mask_R_padded = cv2.copyMakeBorder(contour_mask_R,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            reduced_contour_mask_L = contour_mask_L_padded[cyL+int(border_size/2-crop_size/2):cyL+int(3*border_size/2 + crop_size/2),cxL-border_size:cxL+crop_size + border_size]
            reduced_contour_mask_R = contour_mask_R_padded[cyL+int(border_size/2-crop_size/2):cyL+int(3*border_size/2 + crop_size/2),cxL-border_size:cxL+crop_size + border_size]
                #reduced contour mask is of height crop_size+border_size and length crop_size+2*border_size
            
            cv2.imshow('hey',reduced_contour_mask_L*255)
            #make the stripe mask
            stripe_maskL = stripesL * (1 - reduced_contour_mask_L)
            stripe_maskR = stripesR * (1 - reduced_contour_mask_R)
            
            #pad the image and get it to the right size
            frame_norm_L_masked_padded = cv2.copyMakeBorder(frame_norm_L_masked,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            frame_norm_R_masked_padded = cv2.copyMakeBorder(frame_norm_R_masked,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=0)
            frame_norm_L_masked_striped = frame_norm_L_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxL-border_size:cxL+crop_size + border_size] + stripe_maskL
            frame_norm_R_masked_striped = frame_norm_R_masked_padded[cyL+border_size-int(stripesL.shape[0]/2):cyL+border_size+int(stripesL.shape[0]/2),cxL-border_size:cxL+crop_size + border_size] + stripe_maskR
            
            #compute stereo images from each direction
            stereo_image_L = stereo_left.compute(frame_norm_L_masked_striped,frame_norm_R_masked_striped).astype(np.uint8)
            stereo_image_R = stereo_right.compute(frame_norm_R_masked_striped,frame_norm_L_masked_striped).astype(np.uint8)
            
            #crop and resize masked mice and stereo images
            frame_norm_R_cropped = frame_norm_R_masked_padded[cyR-int(crop_size/2)+border_size:cyR+int(crop_size/2)+border_size,cxR-int(crop_size/2)+border_size:cxR+int(crop_size/2)+border_size]
            frame_norm_L_cropped = frame_norm_L_masked_padded[cyL-int(crop_size/2)+border_size:cyL+int(crop_size/2)+border_size,cxL-int(crop_size/2)+border_size:cxL+int(crop_size/2)+border_size]    
    
            stereo_image_L_cropped = stereo_image_L[int(border_size/2):int(border_size/2+crop_size), int(2*border_size - crop_size/2):int(2*border_size + crop_size/2)]
            stereo_image_R_cropped = stereo_image_R[int(border_size/2):int(border_size/2+crop_size), int(2*border_size - crop_size/2)-(cxL-cxR):int(2*border_size + crop_size/2)-(cxL-cxR)]
    
            stereo_image_combined = ((stereo_image_L_cropped + (255 - stereo_image_R_cropped))*(frame_norm_R_cropped>0)*(frame_norm_L_cropped>0)).astype(np.uint8) #crop_size x crop_size stereo image!
    

            #do Gaussian smoothing of combined stereo image
            stereo_image_cropped_combined_for_gauss = stereo_image_combined
            stereo_image_cropped_combined_gauss = cv2.GaussianBlur(stereo_image_cropped_combined_for_gauss,ksize=(final_smoothing_kernel_width,final_smoothing_kernel_width),
                                                                   sigmaX=final_smoothing_kernel_width,sigmaY=final_smoothing_kernel_width)
                        
            #time filtering removed for now...
            
            #flip mouse the right way
            rotate_angle, face_left, ellipse, topright_or_botleft, ellipse_width = \
            flip_mouse(face_left, ellipse, topright_or_botleft, stereo_image_cropped_combined_gauss, sausage_thresh = 1.1)
            
            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1) 
            stereo_image_straight = cv2.warpAffine(stereo_image_cropped_combined_gauss,M,(crop_size,crop_size))        
    
    
            #in case of errors, correct using head=higher and long-term speed=toward head
            stereo_top = stereo_image_straight[0:int(crop_size/2),:]
            stereo_bottom = stereo_image_straight[int(crop_size/2):]
            
            rotate_angle, face_left, depth_ratio, history_x, history_y, x_tip, y_tip, flip = \
            correct_flip(frame_num - start_frame, face_left, stereo_top,stereo_bottom, depth_percentile, depth_ratio, history_x, history_y, cxL, cyL, ellipse, ellipse_width, \
                     width_thresh=width_thresh, speed_thresh=speed_thresh, depth_ratio_thresh = depth_ratio_thresh, pixel_value_thresh = pixel_value_thresh)   
            if flip:
                print('frame ' + str(frame_num-start_frame))
            
            
            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1)
            stereo_image_straight = cv2.warpAffine(stereo_image_cropped_combined_gauss,M,square_size)         
                        
            #save data for further analysis
            if save_data:
                data_video.write(stereo_image_straight)  
                data_times = np.append(data_times,frame_num-1) #minus 1 to put in python coordinates
                mouse_coordinates.append([cxL,cyL])
    
            #prep frames for video presentability
            if write_images or show_images:
                frame_norm_R_resized = cv2.resize(frame_norm_R_cropped,(crop_size*3,crop_size*3))
                frame_norm_L_resized = cv2.resize(frame_norm_L_cropped,(crop_size*3,crop_size*3))            
                stereo_image_L_resized = cv2.resize(stereo_image_L_cropped,(crop_size*3,crop_size*3)) 
                stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(crop_size*3,crop_size*3))    
                stereo_image_combined_resized = cv2.resize(stereo_image_combined,(crop_size*3,crop_size*3))
                stereo_image_combined_resized = cv2.applyColorMap(stereo_image_combined_resized, cv2.COLORMAP_OCEAN)
                cv2.ellipse(stereo_image_cropped_combined_gauss,ellipse,(100,255,100),1)
                cv2.circle(stereo_image_cropped_combined_gauss,(int(ellipse[0][0] + x_tip ),
                                                                int(ellipse[0][1] + y_tip )), 
                                                                radius=3, color=(255,255,255),thickness=5) 
                stereo_image_cropped_combined_gauss_show = cv2.resize(stereo_image_cropped_combined_gauss,(crop_size*3,crop_size*3))
                stereo_image_cropped_combined_gauss_show = cv2.applyColorMap(stereo_image_cropped_combined_gauss_show, cv2.COLORMAP_OCEAN)
                stereo_image_straight = cv2.resize(stereo_image_straight,(crop_size*3,crop_size*3))
                stereo_image_straight = cv2.applyColorMap(stereo_image_straight, cv2.COLORMAP_OCEAN)
                
                
    
            #save videos
            if write_images:
                
                if write_normal_video:
                    frameR = cv2.cvtColor(frame[:,:,r], cv2.COLOR_GRAY2RGB)
                    normal_video.write(frameR)   
                if write_normalized_video:
                    frame_norm_R = cv2.cvtColor(frame_norm_R, cv2.COLOR_GRAY2RGB)
                    normalized_video.write(frame_norm_R)
                if write_cropped_mice:
                    frame_norm_R_resized = cv2.cvtColor(frame_norm_R_resized, cv2.COLOR_GRAY2RGB)
                    cropped_mouse.write(frame_norm_R_resized)
                if write_stereo_inputs:
                    frame_norm_L_masked_striped = cv2.cvtColor(frame_norm_L_masked_striped, cv2.COLOR_GRAY2RGB)
                    stereo_input_L.write(frame_norm_L_masked_striped)
                    frame_norm_R_masked_striped = cv2.cvtColor(frame_norm_R_masked_striped, cv2.COLOR_GRAY2RGB)
                    stereo_input_R.write(frame_norm_R_masked_striped)
                if write_3D_combined:
                    threeD_combined.write(stereo_image_combined_resized)
                if write_3D_smooth: 
                    threeD_smooth.write(stereo_image_cropped_combined_gauss)  
                if write_3D_straight: 
                    threeD_smooth.write(stereo_image_straight)  
          
            #display videos
            
            if show_images:
                if togetherR:
                    twoD_norm_croppedR = '2D_norm_croppedR_together'
                    threeD_combined_gauss = '3D combined_gauss_together'
                    threeD_straight = '3D straight_together'
                elif mouse == 0:
                    twoD_norm_croppedR = '2D_norm_croppedR_mouse1'
                    threeD_combined_gauss = '3D combined_gauss_mouse1'
                    threeD_straight = '3D straight_mouse1'
                if mouse==1:
                    twoD_norm_croppedR = '2D_norm_croppedR_mouse2'
                    threeD_combined_gauss = '3D combined_gauss_mouse2'
                    threeD_straight = '3D straight_mouse2'
                    
    #            cv2.imshow('2D', frame[:,:,r])
                cv2.imshow('2D norm', frame_norm_R)
                cv2.imshow(twoD_norm_croppedR, frame_norm_R_resized)  
                cv2.imshow(threeD_combined_gauss, stereo_image_cropped_combined_gauss_show)
                cv2.imshow(threeD_straight, stereo_image_straight)

        if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
            print(vid.get(cv2.CAP_PROP_POS_FRAMES))
            break
            
        if frame_num >= end_frame:
            break 
        if (frame_num-start_frame)%500==0:
            print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
        
    else:
        print('frame-grabbing problem!!')


vid.release()
if save_data:
    data_video.release()
    np.save(file_loc + '_coordinates', mouse_coordinates)
    np.save(file_loc + '_frames', data_times)

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

np.save(file_loc + 'mouse_array',depth_array)
#cv2.destroyAllWindows()


  