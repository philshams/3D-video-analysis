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

save_vid_name = 'first_run'

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
start_frame = 20400
end_frame = min(np.inf, vid.get(cv2.CAP_PROP_FRAME_COUNT)) #replace np.inf with end frame
l = 1   #left cam is input 1, right cam is input 2...for now
r = 2
mask_thresh = .47 #mouse mask threshold (lower is more stringently darker than background)
kernel = [3,5] #erosion and dilation kernel sizes for mouse mask
iters = [1,2] #erosion and dilation iterations for mouse mask

window_size = 1
min_disparity = 64
num_disparities = 1*16  
smooth_factor = 8
pre_filter_cap = 61 
unique_ratio = 10
max_pixel_diff = 8
final_smoothing_kernel_width = 13
time_filter_width = 2 # 0 or 1 or 2
time_filter_weight = .5 # 0 to 1


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
        
if save_data == True:
    fourcc = cv2.VideoWriter_fourcc(*'LJPG') #LJPG for lossless, MJP2 /MJPG works; try MJP2 or LAGS or 'Y16 '; want uncompressed!!
    data_file = file_loc + 'data.avi'
    if os.path.isfile(data_file) and do_not_overwrite:
        raise Exception('File already exists') 
    data_video = cv2.VideoWriter(data_file,fourcc , frame_rate, (150,150), False) 
 




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

#initialize time-smoothing
if time_filter_width > 0:
    stereo_image_cropped_combined_gauss_prev4 = np.zeros((150,150)).astype(np.uint8)
    stereo_image_cropped_combined_gauss_prev3 = np.zeros((150,150)).astype(np.uint8)
    stereo_image_cropped_combined_gauss_prev2 = np.zeros((150,150)).astype(np.uint8)
    stereo_image_cropped_combined_gauss_prev = np.zeros((150,150)).astype(np.uint8)
    stereo_image_cropped_combined_gauss_cur = np.zeros((150,150)).astype(np.uint8)
    
#initialize head-spotting
face_left = 1
topright_or_botleft = 1
topright_or_botleft_prev = 1
cur_depth_ratio = 1
prev_depth_ratio = 1
top_mask = np.concatenate((np.ones((75,150)),np.zeros((75,150)))).astype(np.uint8)
bottom_mask = np.concatenate((np.zeros((75,150)),np.ones((75,150)))).astype(np.uint8)
cxL = 400
cyL = 240
move_prev = 0

#initialize erosion/dilation kernels
kernel_er = np.ones((kernel[0],kernel[0]),np.uint8)
kernel_dil = np.ones((kernel[1],kernel[1]),np.uint8)
kernel_head = np.ones((9,9),np.uint8)
        
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
        cxL_prev = cxL
        cyL_prev = cyL
        contoursL, big_cnt_indL, cxL, cyL, cntL = get_biggest_contour(frame_norm_L_mask)
        contoursR, big_cnt_indR, cxR, cyR, cntR = get_biggest_contour(frame_norm_R_mask)
        
        #get and apply mask for the biggest contour
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
        contour_mask_R = (cv2.drawContours(blank, contoursR, big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))
        frame_norm_R_masked = frame_norm_R * contour_mask_R
        blank = np.zeros(frame_norm_R.shape).astype(np.uint8)
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
        
        #crop and resize masked mice and stereo images
        frame_norm_R_cropped = frame_norm_R_masked[cyR-75:cyR+75,cxR-75:cxR+75]
        frame_norm_L_cropped = frame_norm_L_masked[cyL-75:cyL+75,cxL-75:cxL+75]    

        stereo_image_L_cropped = stereo_image_L[25:175,125:275]
        stereo_image_R_cropped = stereo_image_R[25:175,125-(cxL-cxR):275-(cxL-cxR)]

        stereo_image_combined = ((stereo_image_L_cropped + (255 - stereo_image_R_cropped))*(frame_norm_R_cropped>0)*(frame_norm_L_cropped>0)).astype(np.uint8)
        
 

        #do time-smoothing
        stereo_image_cropped_combined_gauss_prev4 = stereo_image_cropped_combined_gauss_prev3
        stereo_image_cropped_combined_gauss_prev3 = stereo_image_cropped_combined_gauss_prev2
        stereo_image_cropped_combined_gauss_prev2 = stereo_image_cropped_combined_gauss_prev
        stereo_image_cropped_combined_gauss_prev = stereo_image_cropped_combined_gauss_cur
        
        
        #do Gaussian smoothing of combined stereo image
        stereo_image_cropped_combined_for_gauss = stereo_image_combined
        stereo_image_cropped_combined_gauss_cur = cv2.GaussianBlur(stereo_image_cropped_combined_for_gauss,ksize=(final_smoothing_kernel_width,final_smoothing_kernel_width),
                                                               sigmaX=final_smoothing_kernel_width,sigmaY=final_smoothing_kernel_width)
        
        
        if frame_num < start_frame + 1 +2*time_filter_width and time_filter_width > 0:
            continue
        elif time_filter_width == 0:
            stereo_image_cropped_combined_gauss = stereo_image_cropped_combined_gauss_cur
        elif time_filter_width == 1:
            stereo_image_cropped_combined_gauss = (np.sum(np.array([time_filter_weight*stereo_image_cropped_combined_gauss_cur,stereo_image_cropped_combined_gauss_prev,
                                                                    time_filter_weight*stereo_image_cropped_combined_gauss_prev2]),axis=0)/(1+2*time_filter_weight)).astype(np.uint8)
        elif time_filter_width == 2:
            stereo_image_cropped_combined_gauss = (np.sum(np.array([time_filter_weight*time_filter_weight*stereo_image_cropped_combined_gauss_cur,time_filter_weight*stereo_image_cropped_combined_gauss_prev,
                                                                    stereo_image_cropped_combined_gauss_prev2,time_filter_weight*stereo_image_cropped_combined_gauss_prev3,
                                                                    time_filter_weight*time_filter_weight*stereo_image_cropped_combined_gauss_prev4]),
                                                                    axis=0)/(1+2*time_filter_weight+2*time_filter_weight*time_filter_weight)).astype(np.uint8)   

        #get ellipse of mouse
        _, contours_stereo, _ = cv2.findContours(stereo_image_cropped_combined_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipse = cv2.fitEllipse(contours_stereo[0])
        
        
        #prevent flips of orientation, when the mouse is sufficiently sausage-like
        topright_or_botleft_prev = topright_or_botleft
        
        topright_or_botleft = np.sign(np.cos(np.deg2rad(ellipse[2])))
        if topright_or_botleft_prev != topright_or_botleft and (not 50 < ellipse[2] < 130):
            face_left*=-1
            print('face_swap!')
            

        cols = 150
        rows = 150
        if face_left == 1:
            rotate_angle = (ellipse[2]-180)
        else:
            rotate_angle = ellipse[2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
        stereo_image_straight = cv2.warpAffine(stereo_image_cropped_combined_gauss,M,(cols,rows))        
        
        
        #if (ellipse[1][1] / ellipse[1][0]) < 1.6:
        stereo_top = stereo_image_straight*top_mask
        stereo_bottom = stereo_image_straight*bottom_mask
        major_top_avg = np.percentile(stereo_top[stereo_top>0], 55) #np.median(stereo_top[stereo_top>0])
        major_bottom_avg = np.percentile(stereo_bottom[stereo_bottom>0], 55) #np.median(stereo_bottom[stereo_bottom>0])
        
        #print((major_top_avg))
        #print((major_bottom_avg))
        #print('')
        #print(np.mean((major_top_avg,major_bottom_avg)) )
        #print(ellipse[2])
        #print((major_top_avg / major_bottom_avg))
        #print(ellipse[1][1] / ellipse[1][0])
        ellipse_width = (ellipse[1][1] / ellipse[1][0])
        prev_depth_ratio = cur_depth_ratio
        cur_depth_ratio = major_top_avg / major_bottom_avg
        
        x_tip = face_left*.5*ellipse[1][1]*np.cos(np.deg2rad(ellipse[2]+90))
        y_tip = face_left*.5*ellipse[1][1]*np.sin(np.deg2rad(ellipse[2]+90))
        
        

            
        delta_x = cxL - cxL_prev
        delta_y = cyL - cyL_prev
        speed = np.sqrt(delta_x**2 + delta_y**2)
        heading = [(delta_x+.001) / (speed+.001), (delta_y+.001) / (speed+.001)]
        
        vec_length = np.sqrt(x_tip**2+y_tip**2)
        head_dir_putative = [x_tip/vec_length,-y_tip/vec_length]
        direction_dot = np.dot(heading,head_dir_putative)
        
        if (cur_depth_ratio < .9 and prev_depth_ratio < .9 and ellipse_width > 1 and np.mean((major_top_avg,major_bottom_avg)) > 130 and speed < 3) or (speed > 3 and direction_dot > 0) :
            face_left *= -1
            print('face_swap_correction!')
        #if ellipse_width < .. turn on minor axis? later if necessary
        
        
        
        print('x,y movement')
        print(delta_x)
        print(delta_y)
        print('speed')
        print(speed)
        print('heading')
        print(heading)
        
        print('put head dir')
        print(head_dir_putative)
        
        print('dot_product')
        print(np.dot(heading,head_dir_putative))
        print('')
        
        
        
        
        
        
        cols = 150
        rows = 150
        if face_left == 1:
            rotate_angle = (ellipse[2]-180)
        else:
            rotate_angle = ellipse[2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
        stereo_image_straight = cv2.warpAffine(stereo_image_cropped_combined_gauss,M,(cols,rows))   
            
            
            
            
            
        #minor_left_avg = 
        #minor_right_avg = 
        #print(major_top_avg / major_bottom_avg)
        #print(ellipse[1][1] / ellipse[1][0])

        
        #print(topright_or_botleft)
        #print(ellipse[1][1] / ellipse[1][0])
#if corr[0,1] > .2:  #find better way -- when mouse is vertical this fails -- should be relative length of axes
#--split ellipse in half, along major (and minor depending on axis relative length) axis; expect head side to be greater (closer) on average
                                        #   and thinner (ellipse fitted to each size, more orthogonal in butt and paralell in head)
                                        #  should transform to be straight first! -- to facilitate this analysis

                                        #and if running (cxL/ cyL change sufficiently), use that to get hd
#            
#        else:

#            coord = max_ind - 74
#            hd = -np.sign(coord[0]*coord[1])
#            print('hd')
#            print(hd)
#            if hd_prev != hd:
#                face_left*=-1
#                print('face_left!')
        
        #find head
#        stereo_image_cropped_combined_gauss_blur = stereo_image_cropped_combined_gauss*(stereo_image_cropped_combined_gauss > 160) #cv2.GaussianBlur(255*(stereo_image_cropped_combined_gauss>170).astype(uint8),ksize=(17,17),
#                                                                                               #sigmaX=17,sigmaY=17)
#        stereo_image_cropped_combined_gauss_blur =  cv2.erode(stereo_image_cropped_combined_gauss_blur, kernel_head, iterations=1)                                                                                   
#        cv2.imshow('blur',stereo_image_cropped_combined_gauss_blur)
#        max_ind = np.array(np.unravel_index(stereo_image_cropped_combined_gauss(stereo_image_cropped_combined_gauss.argmin(), 
#                                            stereo_image_cropped_combined_gauss.shape)).astype(np.uint8)                    
#        cv2.circle(stereo_image_cropped_combined_gauss,(max_ind[0], max_ind[1]), 
#                                                        radius=3, color=(255,255,255),thickness=2)        
#        

        
        
        

        
        #print(corr[0,1])
        #print(ellipse[2])
        #print(cxL)
        #print(cyL)
        #print(hd)
        #print(face_left)
        #print(np.cos(np.deg2rad(ellipse[2])))
        #print('')
        #face_left = 1

        
        
        
        
         
        
        if save_data:
            data_video.write(stereo_image_cropped_combined_gauss)  
            #cv2.imshow('test',stereo_image_cropped_combined_gauss)
       
    
        #prep frames for video presentability
        if write_images or show_images:
            frame_norm_R_resized = cv2.resize(frame_norm_R_cropped,(150*3,150*3))
            frame_norm_L_resized = cv2.resize(frame_norm_L_cropped,(150*3,150*3))            
            stereo_image_L_resized = cv2.resize(stereo_image_L_cropped,(150*3,150*3)) 
            stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(150*3,150*3))    
            stereo_image_combined_resized = cv2.resize(stereo_image_combined,(150*3,150*3))
            stereo_image_combined_resized = cv2.applyColorMap(stereo_image_combined_resized, cv2.COLORMAP_OCEAN)
            cv2.ellipse(stereo_image_cropped_combined_gauss,ellipse,(100,255,100),1)
            #print(ellipse[2])
            cv2.circle(stereo_image_cropped_combined_gauss,(int(ellipse[0][0] + x_tip ),
                                                            int(ellipse[0][1] + y_tip )), 
                                                            radius=3, color=(255,255,255),thickness=5) 
            #cv2.imshow('3D combined_gauss', stereo_image_cropped_combined_gauss)
            stereo_image_cropped_combined_gauss_show = cv2.resize(stereo_image_cropped_combined_gauss,(150*3,150*3))
            stereo_image_cropped_combined_gauss_show = cv2.applyColorMap(stereo_image_cropped_combined_gauss_show, cv2.COLORMAP_OCEAN)
            stereo_image_straight = cv2.resize(stereo_image_straight,(150*3,150*3))
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
      
        #display videos
        if show_images:
            
#            cv2.imshow('maskcheckL',frame_norm_L_masked_striped)
#            cv2.imshow('maskcheckR',frame_norm_R_masked_striped)
#            cv2.imshow('2D', frame[:,:,r])
            cv2.imshow('2D_norm', frame_norm_R)
#            cv2.imshow('2D_norm_croppedR', frame_norm_R_resized)
            cv2.imshow('2D_norm_croppedL', frame_norm_L_resized) 
#            cv2.imshow('3D left', stereo_image_L_resized) 
#            cv2.imshow('3D right', stereo_image_R_resized) 
#            cv2.imshow('3D combined', stereo_image_combined_resized)    
            cv2.imshow('3D combined_gauss', stereo_image_cropped_combined_gauss_show)
            cv2.imshow('3D straight', stereo_image_straight)

        
        if show_images:
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
    np.save(file_loc + 'x_coordinate', cxL)
    np.save(file_loc + 'y_coordinate', cyL)

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

#cv2.destroyAllWindows()


  