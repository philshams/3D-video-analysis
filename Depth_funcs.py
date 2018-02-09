# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:21:42 2018

@author: SWC
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

    
def get_background_mean(file_name = '', file_loc = '', avg_over_approx = 10000):
    save_file = file_loc + 'background_mat_avg.npy'
    if os.path.isfile(save_file):
        raise Exception('File already exists') 
    
    vid = cv2.VideoCapture(file_name)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    background_mat = np.zeros((height, width, 2 ))
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    every_other = int(num_frames / avg_over_approx)

    avg_multiplier = (int(num_frames/every_other))-1 #skip the first and last 2 frames and do about avg_over_approx frames
    
    i = 0
    j = 0
    k = 0
    done_marker = np.arange(0,100,10)
    
    print('computing mean background...')
    
    while True:
        i += 1
        ret, frame = vid.read() # get the frame
        if ret and i%every_other==0:
            # store the current frame in as a numpy array
            background_mat[:,:,0] += frame[:,:,1] / avg_multiplier
            background_mat[:,:,1] += frame[:,:,2] / avg_multiplier
            j+= 1
            
            percent_there = int(100*j/avg_multiplier)
            print(j)
            if percent_there%10==0 and percent_there == done_marker[k]:
                print(str(done_marker[k]) + '% done')
                k+=1

        if vid.get(cv2.CAP_PROP_POS_FRAMES)+1 == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is almost equal to the total number of frames, stop
            break
    cv2.imshow('background left cam', background_mat[:,:,0].astype(np.uint8))
    cv2.imshow('background right cam', background_mat[:,:,1].astype(np.uint8))
    
    np.save(save_file,background_mat)
    
    return background_mat
    
    
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


def get_y_offset(file_name,background_mat,start_frame,stop_frame,l=1,r=2, mask_thresh = .5, kernel = [3,5], iters = [1,2]): #kernel and iters for erosion and dilation, respectively
    
    vid = cv2.VideoCapture(file_name)
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