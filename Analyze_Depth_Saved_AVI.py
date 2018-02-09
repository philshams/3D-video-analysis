#%load_ext autoreload
#%autoreload 2

#%%Stereoscopic AVI file to be analyzed
import numpy as np
import cv2
import os
from Depth_funcs import create_global_matchers, get_background_mean, make_striped_background, get_y_offset, get_biggest_contour


# location of stereoscopic video data
file_name = 'mouse0.avi'
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'

file_loc = file_loc + date + mouse_session
file_name = file_loc + file_name
print(file_name)


# get image sizes
vid = cv2.VideoCapture(file_name)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  
frame_rate = 20

# analysis parameters
show_images = True
display_frame_rate = 1000 #1000 makes it as fast as possible
start_frame = 2190
end_frame = min(2400, vid.get(cv2.CAP_PROP_FRAME_COUNT)) #replace np.inf with end frame
l = 1   #left cam is input 1, right cam is input 2...for now
r = 2
mask_thresh = .5 #mouse mask threshold (lower is more stringently darker than background)
kernel = [3,5] #erosion and dilation kernel sizes for mouse mask
iters = [1,4] #erosion and dilation iterations for mouse mask

window_size = 1
min_disparity = 64
num_disparities = 1*16
smooth_factor = 5
pre_filter_cap = 61 
unique_ratio = 10
max_pixel_diff = 8
final_smoothing_kernel_width = 13

write_images = True
do_not_overwrite = False
write_normal_video = True
write_normalized_video = True
write_cropped_mice = True
write_stereo_inputs = True
write_3D_individual = True
write_3D_combined = True
write_3D_smooth = True



# Get one-time parameters
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


# make striped background - in size of ROI for 3D mouse detection
stripesL, stripesR = make_striped_background(height, width, min_disparity, roi_height=200, roi_width=350)
#cv2.imshow('stripe background',stripesR)


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
fourcc = cv2.VideoWriter_fourcc(*'MJPG') #LJPG for lossless, MJP2 /MJPG works; try MJP2 or LAGS or 'Y16 '; want uncompressed!!
file_loc = file_loc + 'Dario_snip_'

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
    if write_3D_individual == True:
        video_file = file_loc + '3D_single_pass.avi'
        if os.path.isfile(video_file) and do_not_overwrite:
            raise Exception('File already exists') 
        threeD_individual = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450))#, False )
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

 




#cv2.destroyAllWindows()
    

#%% Run 3D analyser over frames
    
slope_recipr = 1
square_size = 100

#set up background image
background_L = background_mat[:,:,0]          
background_R = cv2.copyMakeBorder(background_mat[:,:,1],top=y_offset,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
background_R = background_R[:-y_offset,:]    
          
#initialize mouse video
vid = cv2.VideoCapture(file_name)
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        
while True:
    ret, frame = vid.read() # get the frame
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
        kernel_er = np.ones((kernel[0],kernel[0]),np.uint8)
        kernel_dil = np.ones((kernel[1],kernel[1]),np.uint8)
        frame_norm_L_mask = ((frame_norm_L / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_R_mask = ((frame_norm_R / (256/2)) < mask_thresh).astype(np.uint8) 
        frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=iters[0]) 
        frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=iters[0])
        frame_norm_L_mask = cv2.dilate(frame_norm_L_mask, kernel_dil, iterations=iters[1]) 
        frame_norm_R_mask = cv2.dilate(frame_norm_R_mask, kernel_dil, iterations=iters[1])


        #get biggest contour from the left and right masked images
        contoursL, big_cnt_indL, cxL, cyL, cntL = get_biggest_contour(frame_norm_L_mask)
        contoursR, big_cnt_indR, cxR, cyR, cntR = get_biggest_contour(frame_norm_R_mask)
        
        #get and apply mask for the biggest contour
        blank = np.zeros(frame_norm_R.shape).astype(uint8)
        contour_mask_R = (cv2.drawContours(blank, contoursR, big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_R_masked = frame_norm_R * contour_mask_R
        blank = np.zeros(frame_norm_R.shape).astype(uint8)
        contour_mask_L = (cv2.drawContours(blank, contoursL, big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_L_masked = frame_norm_L * contour_mask_L

        #pad the contour map, get it to the right size
        contour_mask_L_padded = cv2.copyMakeBorder(contour_mask_L,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        contour_mask_R_padded = cv2.copyMakeBorder(contour_mask_R,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        reduced_contour_mask_L = contour_mask_L_padded[cyL:cyL+200,cxL-100:cxL+250]
        reduced_contour_mask_R = contour_mask_R_padded[cyL:cyL+200,cxL-100:cxL+250]

        
        #make the stripe mask
        stripe_maskL = stripesL * (1 - reduced_contour_mask_L)
        stripe_maskR = stripesR * (1 - reduced_contour_mask_R)
        
        #pad the image and get it to the right size
        frame_norm_L_masked_padded = cv2.copyMakeBorder(frame_norm_L_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        frame_norm_R_masked_padded = cv2.copyMakeBorder(frame_norm_R_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        frame_norm_L_masked_striped = frame_norm_L_masked_padded[cyL:cyL+200,cxL-100:cxL+250] + stripe_maskL
        frame_norm_R_masked_striped = frame_norm_R_masked_padded[cyL:cyL+200,cxL-100:cxL+250] + stripe_maskR

        #compute stereo images from each direction
        stereo_image_L = stereo_left.compute(frame_norm_L_masked_striped,frame_norm_R_masked_striped).astype(np.uint8)
        stereo_image_R = stereo_right.compute(frame_norm_R_masked_striped,frame_norm_L_masked_striped).astype(np.uint8)
        
        
        #the rest is mostly cosmetics...
        
        #crop and resize masked mice and stereo images
        frame_norm_R_cropped = frame_norm_R_masked[cyR-75:cyR+75,cxR-75:cxR+75]
        frame_norm_L_cropped = frame_norm_L_masked[cyL-75:cyL+75,cxL-75:cxL+75]    

        stereo_image_L_cropped = stereo_image_L[25:175,125:275]
        stereo_image_R_cropped = stereo_image_R[25:175,125-(cxL-cxR):275-(cxL-cxR)]

        stereo_image_combined = ((stereo_image_L_cropped + (255 - stereo_image_R_cropped))*(frame_norm_R_cropped>0)*(frame_norm_L_cropped>0)).astype(np.uint8)
        
        #do Gaussian smoothing of combined stereo image
        stereo_image_cropped_combined_for_gauss = stereo_image_combined
        stereo_image_cropped_combined_gauss = cv2.GaussianBlur(stereo_image_cropped_combined_for_gauss,ksize=(final_smoothing_kernel_width,final_smoothing_kernel_width),
                                                               sigmaX=final_smoothing_kernel_width,sigmaY=final_smoothing_kernel_width)
        
        #get spine slope of mouse
        _, contours_stereo, _ = cv2.findContours(stereo_image_cropped_combined_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        #calculate slope to get spine alignment of mouse
        corr = np.corrcoef(np.squeeze(contours_stereo[0]).T)        
        x = 225
        y = 225
        if np.abs(corr[0,1]) > .031:  #if corr coeff under certain value, use previous slope
            [vxR,vyR,x,y] = cv2.fitLine(cntR, cv2.DIST_L2,0,0.01,0.01)
            [vxL,vyL,x,y] = cv2.fitLine(cntL, cv2.DIST_L2,0,0.01,0.01)
            slope_recipr = np.mean([(vxL/vyL),(vxR/vyR)])

       
    
        #prep frames for video presentability
        if write_images or show_images:
            frame_norm_R_resized = cv2.resize(frame_norm_R_cropped,(150*3,150*3))
            frame_norm_L_resized = cv2.resize(frame_norm_L_cropped,(150*3,150*3))            
            stereo_image_L_resized = cv2.resize(stereo_image_L_cropped,(150*3,150*3)) 
            stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(150*3,150*3))    
            stereo_image_combined_resized = cv2.resize(stereo_image_combined,(150*3,150*3))
            stereo_image_combined_resized = cv2.applyColorMap(stereo_image_combined_resized, cv2.COLORMAP_OCEAN)
            stereo_image_cropped_combined_gauss = cv2.resize(stereo_image_cropped_combined_gauss,(150*3,150*3))
            stereo_image_cropped_combined_gauss = cv2.applyColorMap(stereo_image_cropped_combined_gauss, cv2.COLORMAP_OCEAN)
            #stereo_image_cropped_combined_gauss = cv2.line(stereo_image_cropped_combined_gauss,(int(225-225*(slope_recipr)),0),(int(225+225*(slope_recipr)),450),(100,10,10),3)

        #save videos
        if write_images:
            
            frameR = cv2.cvtColor(frame[:,:,r], cv2.COLOR_GRAY2RGB)
            normal_video.write(frameR)   
            frame_norm_R = cv2.cvtColor(frame_norm_R, cv2.COLOR_GRAY2RGB)
            normalized_video.write(frame_norm_R)
            frame_norm_R_resized = cv2.cvtColor(frame_norm_R_resized, cv2.COLOR_GRAY2RGB)
            cropped_mouse.write(frame_norm_R_resized)
            frame_norm_L_masked_striped = cv2.cvtColor(frame_norm_L_masked_striped, cv2.COLOR_GRAY2RGB)
            stereo_input_L.write(frame_norm_L_masked_striped)
            frame_norm_R_masked_striped = cv2.cvtColor(frame_norm_R_masked_striped, cv2.COLOR_GRAY2RGB)
            stereo_input_R.write(frame_norm_R_masked_striped)
            stereo_image_L_resized = cv2.cvtColor(stereo_image_L_resized, cv2.COLOR_GRAY2RGB)
            threeD_individual.write(stereo_image_L_resized)
            threeD_combined.write(stereo_image_combined_resized)
            threeD_smooth.write(stereo_image_cropped_combined_gauss)   
      
        #display videos
        if show_images:
            
            cv2.imshow('maskcheckL',frame_norm_L_masked_striped)
            cv2.imshow('maskcheckR',frame_norm_R_masked_striped)
            cv2.imshow('2D', frame[:,:,r])
            cv2.imshow('2D_norm', frame_norm_R)
            cv2.imshow('2D_norm_croppedR', frame_norm_R_resized)
            cv2.imshow('2D_norm_croppedL', frame_norm_L_resized) 
            cv2.imshow('3D left', stereo_image_L_resized) 
            cv2.imshow('3D right', stereo_image_R_resized) 
            cv2.imshow('3D combined', stereo_image_combined_resized)    
            cv2.imshow('3D combined_gauss', stereo_image_cropped_combined_gauss)
            

        
        if show_images:
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                print(vid.get(cv2.CAP_PROP_POS_FRAMES))
                break
        if vid.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break 
    else:
        print('frame-grabbing problem!!')


vid.release()

if write_images:
    normal_video.release()   
    normalized_video.release()
    cropped_mouse.release()
    stereo_input_L.release() 
    stereo_input_R.release() 
    threeD_individual.release()
    threeD_combined.release()
    threeD_smooth.release()

#cv2.destroyAllWindows()


  