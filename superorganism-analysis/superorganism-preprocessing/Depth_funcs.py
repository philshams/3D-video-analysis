# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:21:42 2018

@author: SWC
%load_ext autoreload
%autoreload 2
"""

import cv2
import numpy as np
import os


def create_global_matchers(window_size = 1, min_disparity = 64, num_disparities = 1*16, smooth_factor = 6, pre_filter_cap = 61, unique_ratio = 10, max_pixel_diff = 8):
    
    stereo_left = cv2.StereoSGBM_create(minDisparity = min_disparity,
    numDisparities = num_disparities,
    blockSize = window_size,
    P1 = smooth_factor*8*window_size**2,  #8
    P2 = smooth_factor*32*window_size**2,  #32
    disp12MaxDiff = max_pixel_diff ,  
    uniquenessRatio = unique_ratio,
#    speckleWindowSize = 200, 
#    speckleRange = 31,
    preFilterCap = pre_filter_cap,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)    #_HH or _SGBM_3WAY`; try BM?
    
    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
    
    shift_pixels = num_disparities+min_disparity
    
    return stereo_left, stereo_right, min_disparity, shift_pixels


#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_background_mean(vid, start_frame = 0, file_loc = '', avg_over = 100):
    save_file = file_loc + '_background_mat_avg.npy'
#    if os.path.isfile(save_file):
#        raise Exception('File already exists') 
    

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    background_mat = np.zeros((height, width, 2 ))
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    every_other = int((num_frames-start_frame) / avg_over)
    
    i = 0
    j = 0
    
    print('computing mean background...')
    
    while True:
        i += 1
        
        if i%every_other==0:
#            print(i)
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read() # get the frame
            
            if ret:
                # store the current frame in as a numpy array
#                print('frame read')
                background_mat[:,:,0] += frame[:,:,1] 
                background_mat[:,:,1] += frame[:,:,2]
                j+= 1
                
                if j >= avg_over: 
                    break    
                
                if avg_over > 200:
                    if j%100 == 0:
                        print(str(j) + ' frames out of ' + str(avg_over) + ' done')
                else:
                    if j%10 == 0:
                        print(str(j) + ' frames out of ' + str(avg_over) + ' done')
            else:
                print('not reading video')


        if vid.get(cv2.CAP_PROP_POS_FRAMES)+1 == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is almost equal to the total number of frames, stop
            break
        
    background_mat = background_mat / j
#    cv2.imshow('background left cam', background_mat[:,:,0].astype(np.uint8))
#    cv2.imshow('background right cam', background_mat[:,:,1].astype(np.uint8))
#    cv2.waitKey(5000)
#    cv2.destroyAllWindows
    np.save(save_file,background_mat)

    
    return background_mat


#    
#def get_background_mean(file_name = '', file_loc = '', avg_over_approx = 1000):
#    save_file = file_loc + 'background_mat_avg.npy'
#    if os.path.isfile(save_file):
#        raise Exception('File already exists') 
#    
#    vid = cv2.VideoCapture(file_loc + file_name)
#    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))    
#    background_mat = np.zeros((height, width, 2 ))
#    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
#       
#    every_other = int(num_frames / avg_over_approx)
#
#    avg_multiplier = (int(num_frames/every_other))-1 #skip the first and last 2 frames and do about avg_over_approx frames
#    
#    i = 0
#    j = 0
#    k = 0
#    done_marker = np.arange(0,110,10)
#    
#    print('computing mean background...')
#    
#    while True:
#        i += 1
#        ret, frame = vid.read() # get the frame
#        if ret and i%every_other==0:
#            # store the current frame in as a numpy array
#            background_mat[:,:,0] += frame[:,:,1] / avg_multiplier
#            background_mat[:,:,1] += frame[:,:,2] / avg_multiplier
#            j+= 1
#            
#            percent_there = int(100*j/avg_multiplier)
#            #print(j)
#            if percent_there%10==0 and percent_there == done_marker[k]:
#                print(str(done_marker[k]) + '% done')
#                if done_marker[k] == 100:
#                    break
#                k+=1
#
#        if vid.get(cv2.CAP_PROP_POS_FRAMES)+1 == vid.get(cv2.CAP_PROP_FRAME_COUNT):
#            # If the number of captured frames is almost equal to the total number of frames, stop
#            break
#    cv2.imshow('background left cam', background_mat[:,:,0].astype(np.uint8))
#    cv2.imshow('background right cam', background_mat[:,:,1].astype(np.uint8))
#    
#    np.save(save_file,background_mat)
#    
#    return background_mat
    
    
def make_striped_background(height, width, min_disparity, roi_height, roi_width):
    stripeboardL = np.zeros((height, width)).astype(np.uint8)
    stripeboardR = np.zeros((height, width)).astype(np.uint8)
    stripe_value = 0
    stripe_size = int(.66*min_disparity)
    for x_start in range(25):
        cv2.rectangle(stripeboardL,(x_start*stripe_size,0),(x_start*stripe_size+stripe_size,height),thickness = -1,color = stripe_value)
        cv2.rectangle(stripeboardR,(x_start*stripe_size,0),(x_start*stripe_size+stripe_size,height),thickness = -1,color = stripe_value)
        
        if stripe_value == 0:
            stripe_value = 255
        else:
            stripe_value = 0

    stripesL = stripeboardL[0:roi_height,0:roi_width]
    stripesR = stripeboardR[0:roi_height,0:roi_width]
    
    return stripesL, stripesR


def get_y_offset(vid,background_mat,start_frame,stop_frame,l=1,r=2, mask_thresh = .5, kernel = [3,5], iters = [1,2]): #kernel and iters for erosion and dilation, respectively
    
#    vid = cv2.VideoCapture(file_name)
    y_offsets = np.array([])
    vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    
    while True:
        ret, frame = vid.read() # get the frame
        if ret:
           
            frame_norm_L = (256/2 * frame[:,:,l] / background_mat[:,:,0] ).astype(np.uint8)
            frame_norm_R = (256/2 * frame[:,:,r] / background_mat[:,:,1] ).astype(np.uint8)
                
            kernel_er = np.ones((kernel[0],kernel[0]),np.uint8)
            kernel_dil = np.ones((kernel[1],kernel[1]),np.uint8)
            frame_norm_L_mask = ((frame_norm_L / (256/2)) < .5).astype(np.uint8) 
            frame_norm_R_mask = ((frame_norm_R / (256/2)) < .5).astype(np.uint8) 
            frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=iters[0]) 
            frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=iters[0])
            frame_norm_L_mask = cv2.dilate(frame_norm_L_mask, kernel_dil, iterations=iters[1])
            frame_norm_R_mask = cv2.dilate(frame_norm_R_mask, kernel_dil, iterations=iters[1])
            
            frame_norm_L_masked = frame_norm_L*frame_norm_L_mask
            frame_norm_R_masked = frame_norm_R*frame_norm_R_mask
            
    
            _, contoursL, _ = cv2.findContours(frame_norm_L_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            cont_count = len(contoursL)
            
            big_cnt_indL = 0
            if cont_count > 1:
                areas = np.zeros(cont_count)
                for c in range(cont_count):
                    areas[c] = cv2.contourArea(contoursL[c])
                big_cnt_indL = np.argmax(areas)
                
            cntL = contoursL[big_cnt_indL]
            M = cv2.moments(cntL)
            cxL = int(M['m10']/M['m00'])    
            cyL = int(M['m01']/M['m00'])
            
            
            _, contoursR, _ = cv2.findContours(frame_norm_R_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       
            
            cont_count = len(contoursR)
            
            big_cnt_indR = 0
            if cont_count > 1:
                areas = np.zeros(cont_count)
                for c in range(cont_count):
                    areas[c] = cv2.contourArea(contoursR[c])
                big_cnt_indR = np.argmax(areas)
                
            cntR = contoursR[big_cnt_indR]
            M = cv2.moments(cntR)
            cxR = int(M['m10']/M['m00'])    
            cyR = int(M['m01']/M['m00'])     
            
            y_offsets = np.append(y_offsets,cyL - cyR)
        
            if vid.get(cv2.CAP_PROP_POS_FRAMES) > stop_frame or vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT): # If the number of captured frames is equal to the total number of frames, stop
                break
        else:
            print('frame-grabbing problem!!')
            
    y_offset_mean = np.mean(y_offsets)
    y_offset_std = np.std(y_offsets)
    print('Y offset of ' + str(y_offset_mean) + ' and std of ' + str(y_offset_std) + ' pixels')
    vid.release()    
        
    return int(y_offset_mean)


def get_biggest_contour(frame):
    _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cont_count = len(contours)
    
    big_cnt_ind = 0
    if cont_count > 1:
        areas = np.zeros(cont_count)
        for c in range(cont_count):
            areas[c] = cv2.contourArea(contours[c])
        big_cnt_ind = np.argmax(areas)
        
    cnt = contours[big_cnt_ind]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])    
    cy = int(M['m01']/M['m00'])        
    
    return contours, big_cnt_ind, cx, cy, cnt

def get_second_biggest_contour(frame, single_mouse_thresh, double_mouse_thresh):
    _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cont_count = len(contours)
    
    scd_big_cnt_ind = 0
    big_cnt_ind = 0
    if cont_count > 1:
        areas = np.zeros(cont_count)
        for c in range(cont_count):
            areas[c] = cv2.contourArea(contours[c]) 
        big_cnt_ind = np.argmax(areas)
        biggest_area = areas[big_cnt_ind]
        areas[big_cnt_ind] = 0
        scd_big_cnt_ind = np.argmax(areas)
        second_biggest_area = areas[scd_big_cnt_ind]
        
#        print('biggest area is ' + str(biggest_area))
#        print('2nd biggest area is ' + str(second_biggest_area))
    if big_cnt_ind == scd_big_cnt_ind:
        together = True
    elif biggest_area < double_mouse_thresh and second_biggest_area > single_mouse_thresh:
        together = False
    else:
        together = True
    
        
    cnt2 = contours[scd_big_cnt_ind]
    M = cv2.moments(cnt2)
    cx2 = int(M['m10']/M['m00'])    
    cy2 = int(M['m01']/M['m00'])        
    
    return scd_big_cnt_ind, cx2, cy2, cnt2, together   
    
def flip_mouse(face_left, ellipse, topright_or_botleft, stereo_image_cropped_combined_gauss, sausage_thresh = 1.1):
    #get ellipse of mouse
    prev_ellipse = ellipse
    topright_or_botleft_prev = topright_or_botleft

    _, contours_stereo, _ = cv2.findContours(stereo_image_cropped_combined_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
    
    ellipse = cv2.fitEllipse(contours_stereo[0])
    ellipse_width = (ellipse[1][1] / ellipse[1][0])
    if ellipse_width < sausage_thresh:
        ellipse = prev_ellipse   

    #prevent flips of orientation, when the mouse is sufficiently sausage-like
    topright_or_botleft = np.sign(np.cos(np.deg2rad(ellipse[2])))
    if topright_or_botleft_prev != topright_or_botleft and (not 50 < ellipse[2] < 130):
        face_left*=-1
        #print('face_swap!') 
        
    #rotate the mouse accordingly    
    if face_left == 1:
        rotate_angle = (ellipse[2]-180)
    else:
        rotate_angle = ellipse[2]
    
    return rotate_angle, face_left, ellipse, topright_or_botleft, ellipse_width


def correct_flip(initiation, face_left, stereo_top,stereo_bottom, depth_percentile, depth_ratio, history_x, history_y, cxL, cyL, ellipse, ellipse_width, \
                 width_thresh=1.2, speed_thresh=12, depth_ratio_thresh = [.8, .7, .6], pixel_value_thresh = [109,89,69]):
    #calculate depth image stats
    major_top_avg = np.percentile(stereo_top[stereo_top>0], depth_percentile) 
    major_bottom_avg = np.percentile(stereo_bottom[stereo_bottom>0], depth_percentile) 
    
    depth_ratio[0:2] = depth_ratio[1:3]
    depth_ratio[2] = major_top_avg / major_bottom_avg
    
    x_tip = face_left*.5*ellipse[1][1]*np.cos(np.deg2rad(ellipse[2]+90))
    y_tip = face_left*.5*ellipse[1][1]*np.sin(np.deg2rad(ellipse[2]+90))
     
    flip = 0
    
    history_x[0:3] = history_x[1:4]
    history_y[0:3] = history_y[1:4]
    history_x[3] = cxL
    history_y[3] = cyL
    
    delta_x = cxL - history_x[2] 
    delta_y = -cyL + history_y[2]
    prev_delta_x = history_x[2] - history_x[1] 
    prev_delta_y = -history_y[2] + history_y[1]
#    prev_prev_delta_x = history_y[1] - history_x[0] 
#    prev_prev_delta_y = -history_y[1] + history_y[0]
    
    vec_length = np.sqrt(x_tip**2+y_tip**2)
    head_dir = [x_tip/vec_length,-y_tip/vec_length]
    vel_along_head_dir = np.dot([delta_x,delta_y],head_dir)  
    prev_vel_along_head_dir = np.dot([prev_delta_x,prev_delta_y],head_dir)  


#    if ellipse_width > width_thresh and speed < speed_thresh and ( (np.all(depth_ratio < depth_ratio_thresh[0]) and major_bottom_avg > pixel_value_thresh[0]) \
#                                               or (np.all(depth_ratio < depth_ratio_thresh[1]) and major_bottom_avg > pixel_value_thresh[1]) \
#                                               or (np.all(depth_ratio < depth_ratio_thresh[2]) and major_bottom_avg > pixel_value_thresh[2])):
        
    if ellipse_width > width_thresh and ( (np.median(depth_ratio) < depth_ratio_thresh[0] and major_bottom_avg > pixel_value_thresh[0]) \
                                               or (np.median(depth_ratio) < depth_ratio_thresh[1] and major_bottom_avg > pixel_value_thresh[1]) \
                                               or (np.median(depth_ratio) < depth_ratio_thresh[2] and major_bottom_avg > pixel_value_thresh[2])):
        face_left *= -1
        flip = 1
        print('face_HEIGHT_correction!')
        print(depth_ratio)
        if major_bottom_avg < pixel_value_thresh[0]:
            print('at low pixel value' + str(major_bottom_avg))
        print('')

    elif vel_along_head_dir < -speed_thresh and prev_vel_along_head_dir < -speed_thresh and ellipse_width > width_thresh+.2  and initiation > 7: #speed here 50 works but always off
        face_left *= -1
        flip = 1
        print('face_SPEED_correction!')
       

    if face_left == 1:
        rotate_angle = (ellipse[2]-180)
    else:
        rotate_angle = ellipse[2]
        
    x_tip = face_left*.5*ellipse[1][1]*np.cos(np.deg2rad(ellipse[2]+90))
    y_tip = face_left*.5*ellipse[1][1]*np.sin(np.deg2rad(ellipse[2]+90))   
    
    return rotate_angle, face_left, depth_ratio, history_x, history_y, x_tip, y_tip, flip


def write_videos(file_loc, write_images, do_not_overwrite, fourcc, frame_rate, width, height, crop_size, border_size,
                 write_normal_video, write_normalized_video, write_cropped_mice, write_stereo_inputs, write_3D_combined, write_3D_smooth):
        
    normal_video, normalized_video, cropped_mouse, stereo_input_L, stereo_input_R, threeD_combined, threeD_smooth = np.zeros(7)
    
    if write_images == True:
        if write_normal_video == True:
            video_file = file_loc + '_normal_video.avi'
            if os.path.isfile(video_file) and do_not_overwrite:
                raise Exception('File already exists') 
            normal_video = cv2.VideoWriter(video_file,fourcc , frame_rate, (width,height))#, False) 
        if write_normalized_video == True:
            video_file = file_loc + '_normalized_video.avi'
            if os.path.isfile(video_file) and do_not_overwrite:
                raise Exception('File already exists')         
            normalized_video = cv2.VideoWriter(video_file,fourcc , frame_rate, (width,height))#, False) 
        if write_cropped_mice == True:
            video_file = file_loc + '_cropped_mouse.avi'
            if os.path.isfile(video_file) and do_not_overwrite:
                raise Exception('File already exists')         
            cropped_mouse = cv2.VideoWriter(video_file,fourcc , frame_rate, (3*crop_size,3*crop_size))#, False) 
        if write_stereo_inputs == True:
            video_fileL = file_loc + '_stereo_input_L.avi'
            video_fileR = file_loc + '_stereo_input_R.avi'
            if (os.path.isfile(video_fileL) or os.path.isfile(video_fileR)) and do_not_overwrite:
                raise Exception('File already exists') 
            stereo_input_L = cv2.VideoWriter(video_fileL,fourcc , frame_rate, (crop_size + 2*border_size,crop_size + border_size))#, False) 
            stereo_input_R = cv2.VideoWriter(video_fileR,fourcc , frame_rate, (crop_size + 2*border_size,crop_size + border_size))#, False) 
        if write_3D_combined == True:
            video_file = file_loc + '_3D_combined.avi'
            if os.path.isfile(video_file) and do_not_overwrite:
                raise Exception('File already exists')         
            threeD_combined = cv2.VideoWriter(video_file,fourcc , frame_rate, (3*crop_size,3*crop_size))#, True) 
        if write_3D_smooth == True:
            video_file = file_loc + '_3D_smooth.avi'
            if os.path.isfile(video_file) and do_not_overwrite:
                raise Exception('File already exists') 
            threeD_smooth = cv2.VideoWriter(video_file,fourcc , frame_rate, (3*crop_size,3*crop_size))#, True)
    
    return normal_video, normalized_video, cropped_mouse, stereo_input_L, stereo_input_R, threeD_combined, threeD_smooth
        
    
    
    
#local matcher code:
#stereo_left = cv2.StereoBM_create()
#stereo_left.setMinDisparity(64)
#stereo_left.setNumDisparities(num_disparities)
#stereo_left.setBlockSize(SADws)
##stereo_left.setDisp12MaxDiff(50)  #50
##stereo_left.setUniquenessRatio(1)  #10
##stereo_left.setPreFilterSize(5)
##stereo_left.setPreFilterCap(25)
#stereo_left.setTextureThreshold(500)
##stereo_left.setSpeckleWindowSize(200) #or off?
##stereo_left.setSpeckleRange(31)  #or off?
#stereo_left.setPreFilterType(0) #STEREO_BM_XSOBEL; 1 may be   STEREO BM NORMALIZED RESPONSE)
    
#weight least squares filter code:
#wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
#wls_filter.setLambda(lmbda)
#wls_filter.setSigmaColor(sigma)   
#wls_filter.setDepthDiscontinuityRadius(disc_rad)
    
#ROI code:
#    select_roi = False
#if select_roi:
#    ret, frame = vid.read()
#    roi = cv2.selectROI(frame)
#    roi = np.array(roi).astype(int)
#else:
#    try:
#        roi[3]
#        if use_roi == False:
#            roi = [0,0,width,height]
#    except:
#        roi = [0,0,width,height]
#        
#background_mat = background_mat[roi[1]:roi[1]+roi[3], :, :]
    
#put square around roi
        #extract square around mouse
#        blank = np.zeros(frame_norm_L.shape).astype(np.uint8)
#        frame_norm_L_mask_square = cv2.rectangle(blank,(cxL-square_size,cyL-square_size),(cxL+square_size,cyL+square_size),thickness = -1,color = 1)
#        blank = np.zeros(frame_norm_L.shape).astype(np.uint8)
#        frame_norm_R_mask_square = cv2.rectangle(blank,(cxR-square_size,cyR-square_size),(cxR+square_size,cyR+square_size),thickness = -1,color = 1)
#        
#        frame_norm_L_masked = frame_norm_L * frame_norm_L_mask_square
#        frame_norm_R_masked = frame_norm_R * frame_norm_R_mask_square
#        
#        frame_norm_L_masked2 = frame_L * frame_norm_L_mask_square
#        frame_norm_R_masked2 = frame_R_shift * frame_norm_R_mask_square
        
#        stereo_image_L = stereo_left.compute(frame_norm_L_masked2,frame_norm_R_masked2).astype(np.uint8)
#        stereo_image_R = stereo_right.compute(frame_norm_R_masked2,frame_norm_L_masked2).astype(np.uint8)

#code for circle drawing
#cv2.circle(frame_norm_R_masked, (cxR, cyR), radius=3, color=(255,255,255),thickness=5) 
#cv2.circle(stereo_image_filtered_masked, (int((cxR+cxL)/2), int((cyR+cyL)/2)), radius=3, color=(255,255,255),thickness=3) 
  
#code for angle analysis
#print('L angle = ' + str(np.arctan(vyL/vxL)) + 'R angle = ' + str(np.arctan(vyR/vxR)))
#if (np.arctan(vyL/vxL) - np.arctan(vyR/vxR)) < 1.3: #less than 75 deg off
        
#code for ellipse over mouse
        #        #get spine slope of mouse
#        _, contours_stereo, _ = cv2.findContours(stereo_image_cropped_combined_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#       
#        #calculate slope to get spine alignment of mouse
#        corr = np.corrcoef(np.squeeze(contours_stereo[0]).T)        
#        x = 225
#        y = 225
#        if np.abs(corr[0,1]) > .031:  #if corr coeff under certain value, use previous slope
#            [vxR,vyR,x,y] = cv2.fitLine(cntR, cv2.DIST_L2,0,0.01,0.01)
#            [vxL,vyL,x,y] = cv2.fitLine(cntL, cv2.DIST_L2,0,0.01,0.01)
#            slope_recipr = np.mean([(vxL/vyL),(vxR/vyR)])
        #stereo_image_cropped_combined_gauss = cv2.line(stereo_image_cropped_combined_gauss,(int(225-225*(slope_recipr)),0),(int(225+225*(slope_recipr)),450),(100,10,10),3)
        
        
        #head-dir debug prints
#        print('head-up stats')
#        print(depth_ratio)
#        print(major_bottom_avg)
#        print(major_top_avg)
#        print(ellipse_width)       
#        print('x,y movement')
#        print(delta_x)
#        print(delta_y)
#        print('speed')
#        print(speed)
#        print('heading')
#        print(heading)
#        
#        print('put head dir')
#        print(head_dir_putative)
#        
#        print('dot_product')
#        print(np.dot(heading,head_dir_putative))
#        print('')