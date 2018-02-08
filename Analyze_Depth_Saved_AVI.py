# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:57:38 2018

@author: SWC
"""

# get depth of saved video
import numpy as np
import matplotlib.pyplot as plt
import cv2


file = 'mouse0'
file_name = file + '.avi'

background_file = 'background_pre0'
background_file_name = background_file + '.avi'

frame_rate = 60
width = 752
height = 480

num_disparities = 1*16
SADws = 1 #or 3? or 1?
min_disparity = 64
smooth_factor = 6

# FILTER Parameters
lmbda = 8000    #1000
sigma = 1.5     #1.8
disc_rad = 0


#left cam is input 1, right cam is input 2...for now
l = 1
r = 2

get_background_subtract = False
select_roi = True
background_subtract = True
mask = True
use_roi = False

if get_background_subtract:
    
    #%% get background image for subtraction - from video  itself
    
    vid = cv2.VideoCapture(file_name)
    background_mat = np.zeros((height, width, 2 ))
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    avg_multiplier = 1 / (int(num_frames/10)) #skip the first and last 2 frames and do every 10 frames (or more?)
    i = 0
    j = 0
    while True:
        i += 1
        pos_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = vid.read() # get the frame
        if ret and i%10==0:
            # store the current frame in as a numpy array
            background_mat[:,:,0] += frame[:,:,1]*avg_multiplier
            background_mat[:,:,1] += frame[:,:,2]*avg_multiplier
            j+= 1
            print(j)
        if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames, stop
            break
    cv2.imshow('background left cam', background_mat[:,:,0].astype(uint8))
    cv2.imshow('background right cam', background_mat[:,:,1].astype(uint8))
    np.save('background_mat_avg',background_mat)

background_mat = np.load('background_mat_avg.npy')


#%% Create stereo matcher and wls filter
vid = cv2.VideoCapture(file_name)
#image_array = np.zeros((height, width, 2, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))).astype(uint8)
#stereo_array = np.zeros((height, width - num_disparities, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))).astype(uint8)
color_array = np.zeros((height, width - num_disparities, 3)).astype(uint8)

#stereo_left = cv2.StereoSGBM_create(minDisparity = 0,
#    numDisparities = num_disparities,
#    blockSize = SADws)  #used in the code to generate stereo_right and wls_filter



stereo_left = cv2.StereoSGBM_create(minDisparity = min_disparity,
    numDisparities = num_disparities,
    blockSize = SADws,
    P1 = smooth_factor*8*SADws**2,  #8
    P2 = smooth_factor*32*SADws**2,  #32
    disp12MaxDiff = 8,  #50
    uniquenessRatio = 10,  #10
#    speckleWindowSize = 200, #or off?
#    speckleRange = 31,  #or off?
    preFilterCap = 61,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,    #_HH or _SGBM_3WAY`; try BM?
        )

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


stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)

wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)   
wls_filter.setDepthDiscontinuityRadius(disc_rad)



#%% Run 3D analyser over frames
background_mat = np.load('background_mat_avg.npy')
vid = cv2.VideoCapture(file_name)
select_roi = False
if select_roi:
    ret, frame = vid.read()
    roi = cv2.selectROI(frame)
    roi = np.array(roi).astype(int)
else:
    try:
        roi[3]
        if use_roi == False:
            roi = [0,0,width,height]
    except:
        roi = [0,0,width,height]
    
slope_recipr = 1
background_mat = background_mat[roi[1]:roi[1]+roi[3], :, :]
shift_pixels = num_disparities+min_disparity
#fourcc = cv2.VideoWriter_fourcc(*'MJPG') #MJPG works; try MJP2 or LAGS or 'Y16 '
#out = cv2.VideoWriter('test_subtracted_background.avi',fourcc , 10, (450,450))  #0x20363159 corresponds to the Y16 4cc 
checkerboardL = np.zeros((height, width)).astype(uint8)
checkerboardR = np.zeros((height, width)).astype(uint8)
checker_valueL = 0
checker_valueR = 255
for x_start in range(25):
    for y_start in range(1):
        cv2.rectangle(checkerboardL,(x_start*40,y_start*600),(x_start*40+40,y_start*600+600),thickness = -1,color = checker_valueL)
        cv2.imshow('checkerboardL',checkerboardL) 
        cv2.rectangle(checkerboardR,(x_start*40,y_start*600-(shift_pixels-1)),(x_start*40+40,y_start*600+600-(shift_pixels-1)),thickness = -1,color = checker_valueL)
        cv2.imshow('checkerboardR',checkerboardR) 
        
        if checker_valueL == 0:
            checker_valueL = 255
            checker_valueR = 0
        else:
            checker_valueL = 0
            checker_valueR = 255
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
reduced_checkerboardL = checkerboardL[0:200,0:350]
reduced_checkerboardR = checkerboardR[0:200,0:350]
cv2.imshow('checkerboardL',reduced_checkerboardL) 
cv2.imshow('checkerboardR',reduced_checkerboardR) 

            
while True:
#for i in [1]:
    pos_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = vid.read() # get the frame
    if pos_frame<100:
        #print(pos_frame)
        continue
    elif pos_frame>10000:
        break
    if ret:
        # store the current frame in as a numpy array
        #image_array[:,:,1:2,pos_frame] = frame[:,:,1:2]
#        frame_norm_L = (frame[:,:,l] - background_mat[:,:,0] + 256/2).astype(uint8)
#        frame_norm_R = (frame[:,:,r] - background_mat[:,:,1] + 256/2).astype(uint8)
        #crop image
        frame = frame[roi[1]:roi[1]+roi[3], :, :]
        
        frame_norm_L2 = frame[:,:,l]
        frame_norm_R2 = frame[:,:,r]
        frame_norm_R2 = cv2.copyMakeBorder(frame_norm_R2,top=9,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
        frame_norm_R2 = frame_norm_R2[:-9,:]
        
        if background_subtract == True:
            frame_norm_L = (256/2 * frame[:,:,l] / background_mat[:,:,0] ).astype(uint8)
            frame_norm_R = (256/2 * frame[:,:,r] / background_mat[:,:,1] ).astype(uint8)
            frame_norm_R = cv2.copyMakeBorder(frame_norm_R,top=9,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
            frame_norm_R = frame_norm_R[:-9,:]
        else:
            frame_norm_L = frame[:,:,l]
            frame_norm_R = frame[:,:,r]
            frame_norm_R = cv2.copyMakeBorder(frame_norm_R,top=9,bottom=0,left=0,right=0,borderType= cv2.BORDER_REPLICATE,value=0)
            frame_norm_R = frame_norm_R[:-9,:]
            
        if mask:
            kernel_er = np.ones((3,3),np.uint8)
            kernel_dil = np.ones((5,5),np.uint8)
            frame_norm_L_mask = ((frame_norm_L / (256/2)) < .5).astype(uint8) #cv2.dilate(((frame_norm_L / (256/2)) < .5).astype(uint8), None, iterations=0) #.65 and 3 or .55 and 1
            frame_norm_R_mask = ((frame_norm_R / (256/2)) < .5).astype(uint8) #cv2.dilate(((frame_norm_R / (256/2)) < .5).astype(uint8), None, iterations=0)
            frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=1) #.65 and 3 or .55 and 1
            frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=1)
            frame_norm_L_mask = cv2.dilate(frame_norm_L_mask, kernel_dil, iterations=2) #.65 and 3 or .55 and 1
            frame_norm_R_mask = cv2.dilate(frame_norm_R_mask, kernel_dil, iterations=2)
#            frame_norm_L_mask = cv2.erode(frame_norm_L_mask, kernel_er, iterations=1) #.65 and 3 or .55 and 1
#            frame_norm_R_mask = cv2.erode(frame_norm_R_mask, kernel_er, iterations=1)
            
            
            frame_norm_L_masked = frame_norm_L*frame_norm_L_mask
            frame_norm_R_masked = frame_norm_R*frame_norm_R_mask
            
            stereo_mask = (frame_norm_L_mask[:,shift_pixels:]>0)#+(frame_norm_R_mask[:,0:-shift_pixels]>0)
            
#        frame_norm_L= frame[:,:,l]
#        frame_norm_R = frame[:,:,r]
        #rect_L = cv2.remap(frame[:,:,l], mapxL, mapyL,cv2.INTER_LINEAR) # will i need to switch L and R? 
        #rect_R = cv2.remap(frame[:,:,r], mapxR, mapyR,cv2.INTER_LINEAR)
        
#        stereo_image_L = stereo_left.compute(frame_norm_L_masked,frame_norm_R_masked).astype(uint8)
#        stereo_image_R = stereo_right.compute(frame_norm_R_masked,frame_norm_L_masked).astype(uint8)
# 
#        stereo_image_filtered = wls_filter.filter(stereo_image_L, frame_norm_L_masked, None, stereo_image_R) 
#        stereo_image_filtered = stereo_image_filtered[:,shift_pixels:]
#        stereo_image_filtered_masked = stereo_image_filtered*stereo_mask
        
                #raw_disp_vis = cv2.ximgproc.getDisparityVis(stereo_image_L,None,5)
           #raw_disp_vis[:,num_disparities:])
        #cv2.imshow('3D right', stereo_image_R[:,num_disparities:])
                #filtered_disp_vis = cv2.getDisparityVis(stereo_image_filtered,None,5)
#        stereo_image_filtered_color = cv2.applyColorMap(stereo_image_filtered[:,num_disparities:], cv2.COLORMAP_JET)
#        cv2.imshow('3D filtered', stereo_image_filtered_color)

        _, contoursL, _ = cv2.findContours(frame_norm_L_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #or , 1, 2) ?
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
        
        
        _, contoursR, _ = cv2.findContours(frame_norm_R_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #or , 1, 2) ?        
        
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
        
        #print(cyL - cyR)
        
#        cv2.imshow('maskL',contour_mask_L*250)
#        cv2.imshow('maskR',contour_mask_R*250)
#        
        
        #extract square around mouse
        blank = np.zeros(frame_norm_L.shape).astype(uint8)
        frame_norm_L_mask_square = cv2.rectangle(blank,(cxL-100,cyL-100),(cxL+100,cyL+100),thickness = -1,color = 1)
        blank = np.zeros(frame_norm_L.shape).astype(uint8)
        frame_norm_R_mask_square = cv2.rectangle(blank,(cxR-100,cyR-100),(cxR+100,cyR+100),thickness = -1,color = 1)
        
        frame_norm_L_masked = frame_norm_L * frame_norm_L_mask_square
        frame_norm_R_masked = frame_norm_R * frame_norm_R_mask_square
        
        frame_norm_L_masked2 = frame_norm_L2 * frame_norm_L_mask_square
        frame_norm_R_masked2 = frame_norm_R2 * frame_norm_R_mask_square
        
#        stereo_image_L = stereo_left.compute(frame_norm_L_masked2,frame_norm_R_masked2).astype(uint8)
#        stereo_image_R = stereo_right.compute(frame_norm_R_masked2,frame_norm_L_masked2).astype(uint8)

       
        
        #stereo_array[:,:,pos_frame] = stereo_image_filtered[:,num_disparities:]    
        
        blank = np.zeros(frame_norm_R_masked.shape).astype(uint8)
        contour_mask_R = (cv2.drawContours(blank, contoursR, big_cnt_indR, color=(1,1,1), thickness=cv2.FILLED))#.astype(bool)
        frame_norm_R_masked *= contour_mask_R
        blank = np.zeros(frame_norm_R_masked.shape).astype(uint8)
        contour_mask_L = (cv2.drawContours(blank, contoursL, big_cnt_indL, color=(1,1,1), thickness=cv2.FILLED))#.astype(bool)
        frame_norm_L_masked *= contour_mask_L
        #stereo_mask_part_2 = contour_mask_L[:,int(shift_pixels):] #+ contour_mask_R[:,0:-shift_pixels]
        #stereo_image_filtered_masked = stereo_image_filtered_masked * stereo_mask_part_2
        #cv2.circle(stereo_image_filtered_masked, (int((cxR+cxL)/2), int((cyR+cyL)/2)), radius=3, color=(255,255,255),thickness=7) 
        
        
#        stereo_image_filtered = wls_filter.filter(stereo_image_L, frame_norm_L_masked, None, stereo_image_R) 
#        stereo_image_filtered = stereo_image_filtered[:,shift_pixels:]*5
#        stereo_image_filtered_masked = stereo_image_filtered*stereo_mask*2
        
        checkerboard_maskL = checkerboardL * (1 - contour_mask_L) * frame_norm_L_mask_square
        frame_norm_L_masked3 = frame_norm_L_masked + checkerboard_maskL
#        cv2.imshow('maskcheckL',frame_norm_L_masked3)
        checkerboard_maskR = checkerboardR * (1 - contour_mask_R) * frame_norm_R_mask_square
        frame_norm_R_masked3 = frame_norm_R_masked + checkerboard_maskR
#        cv2.imshow('maskcheckR',frame_norm_R_masked3)
        
        contour_mask_L_padded = cv2.copyMakeBorder(contour_mask_L,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        contour_mask_R_padded = cv2.copyMakeBorder(contour_mask_R,100,100,100,100,cv2.BORDER_CONSTANT,value=0)

        reduced_contour_mask_L = contour_mask_L_padded[cyL-100+100:cyL+100+100,cxL-200+100:cxL+150+100]
        reduced_contour_mask_R = contour_mask_R_padded[cyL-100+100:cyL+100+100,cxL-200+100:cxL+150+100]
        reduced_checkerboard_maskL = reduced_checkerboardL * (1 - reduced_contour_mask_L)
        reduced_checkerboard_maskR = reduced_checkerboardR * (1 - reduced_contour_mask_R)
        
        frame_norm_L_masked_padded = cv2.copyMakeBorder(frame_norm_L_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        frame_norm_R_masked_padded = cv2.copyMakeBorder(frame_norm_R_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)

        cv2.imshow('test reduc', reduced_contour_mask_R*255)
        frame_norm_L_masked3 = frame_norm_L_masked_padded[cyL-100+100:cyL+100+100,cxL-200+100:cxL+150+100] + reduced_checkerboard_maskL
        frame_norm_R_masked3= frame_norm_R_masked_padded[cyL-100+100:cyL+100+100,cxL-200+100:cxL+150+100] + reduced_checkerboard_maskR
        cv2.imshow('maskcheckL',frame_norm_L_masked3)
        cv2.imshow('maskcheckR',frame_norm_R_masked3)
        
        stereo_image_L = stereo_left.compute(frame_norm_L_masked3,frame_norm_R_masked3).astype(uint8)
        stereo_image_R = stereo_right.compute(frame_norm_R_masked3,frame_norm_L_masked3).astype(uint8)

        stereo_image_filtered = wls_filter.filter(stereo_image_L, frame_norm_L_masked, None, stereo_image_R) 
        stereo_image_filtered = stereo_image_filtered[:,shift_pixels:]*5
        stereo_image_filtered_masked = stereo_image_filtered*stereo_mask*2
        
                #raw_disp_vis = cv2.ximgproc.getDisparityVis(stereo_image_L,None,5)
        #cv2.imshow('3D left', stereo_image_L[:,shift_pixels:]) 
        cv2.circle(frame_norm_R_masked, (cxR, cyR), radius=3, color=(255,255,255),thickness=5) 
        #cv2.circle(stereo_image_filtered_masked, (int((cxR+cxL)/2), int((cyR+cyL)/2)), radius=3, color=(255,255,255),thickness=3) 
        
        cv2.imshow('3D filtered',stereo_image_filtered )
        cv2.imshow('3D filtered_mask', stereo_image_filtered_masked*5)
        cv2.imshow('2D', frame[:,:,r])
        cv2.imshow('2D_norm', frame_norm_R)
        cv2.imshow('2D_norm_mask', frame_norm_R_masked2)        
        
        
        #pad image
        frame_norm_R_padded = cv2.copyMakeBorder(frame_norm_R_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        frame_norm_R_cropped = frame_norm_R_padded[cyR+25:cyR+175,cxR+25:cxR+175]
        frame_norm_R_resized = cv2.resize(frame_norm_R_cropped,(150*3,150*3))
        
        frame_norm_L_padded = cv2.copyMakeBorder(frame_norm_L_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        frame_norm_L_cropped = frame_norm_L_padded[cyL+25:cyL+175,cxL+25:cxL+175]
        frame_norm_L_resized = cv2.resize(frame_norm_L_cropped,(150*3,150*3))
        
        cv2.imshow('2D_norm_croppedR', frame_norm_R_resized)
        cv2.imshow('2D_norm_croppedL', frame_norm_L_resized)

#        stereo_image_filtered_masked = stereo_image_filtered * frame_norm_L_mask_square[:,shift_pixels:]
#        stereo_image_filtered_padded = cv2.copyMakeBorder(stereo_image_filtered_masked,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
#        stereo_image_filtered_cropped = stereo_image_filtered_padded[int((cyR+cyL)/2)+25:int((cyR+cyL)/2)+175,int((cxR+cxL)/2)+25:int((cxR+cxL)/2)+175]
#        stereo_image_filtered_zeroed = (stereo_image_filtered_cropped)       #-(np.mean(stereo_image_filtered_cropped)*(150*150)/sum(stereo_mask_part_2)) + 128).astype(uint8)
#        stereo_image_filtered_resized = cv2.resize(stereo_image_filtered_zeroed,(150*3,150*3))
#        
#        stereo_image_filtered_resized = cv2.applyColorMap(stereo_image_filtered_resized, cv2.COLORMAP_OCEAN)
#        
#        cv2.imshow('3D_norm_cropped', stereo_image_filtered_resized)
        
        
        #stereo_image_L_padded = cv2.copyMakeBorder(stereo_image_L,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        #stereo_image_L_cropped = stereo_image_L_padded[cyL:cyL+200,cxL:cxL+200]
        stereo_image_L_cropped = stereo_image_L[100-75:100+75,200-75:200+75]
        stereo_image_L_resized = cv2.resize(stereo_image_L_cropped,(150*3,150*3))       
     
        
        cv2.imshow('3D left', stereo_image_L_resized) 

        
        #stereo_image_R_padded = cv2.copyMakeBorder(stereo_image_R,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
        #stereo_image_R_cropped = stereo_image_R_padded[cyR:cyR+200,cxR:cxR+200]
        #stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(150*3,150*3))       
        stereo_image_R_cropped = stereo_image_R[100-75:100+75,200-(cxL-cxR)-75:200-(cxL-cxR)+75]
        stereo_image_R_resized = cv2.resize(stereo_image_R_cropped,(150*3,150*3))
        cv2.imshow('3D right', stereo_image_R_resized) 
 
        stereo_image_cropped_combined = ((stereo_image_L_resized + (255 - stereo_image_R_resized))*(frame_norm_R_resized>0)*(frame_norm_L_resized>0)).astype(uint8)

        
        
        kernel_width = 17
        stereo_image_cropped_combined_for_gauss = stereo_image_cropped_combined
        stereo_image_cropped_combined_gauss = cv2.GaussianBlur(stereo_image_cropped_combined_for_gauss,ksize=(kernel_width,kernel_width),sigmaX=kernel_width,sigmaY=kernel_width)
        #make background ~ right depth for blurring
        _, contours_stereo, _ = cv2.findContours(stereo_image_cropped_combined_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stereo_image_cropped_combined = cv2.applyColorMap(stereo_image_cropped_combined, cv2.COLORMAP_OCEAN)
        stereo_image_cropped_combined_gauss = cv2.applyColorMap(stereo_image_cropped_combined_gauss, cv2.COLORMAP_OCEAN)
        cv2.imshow('3D combined', stereo_image_cropped_combined)   
        cv2.imshow('3D combined_gauss', stereo_image_cropped_combined_gauss)
        #out.write(stereo_image_cropped_combined_gauss)
        
        #if corr coeff under certain value, use previous slope
        
        corr = np.corrcoef(np.squeeze(contours_stereo[0]).T)
        #print(corr[0,1])
        
        x = 225
        y = 225
        if np.abs(corr[0,1]) > .01: 
            [vxR,vyR,x,y] = cv2.fitLine(cntR, cv2.DIST_L2,0,0.01,0.01)
            [vxL,vyL,x,y] = cv2.fitLine(cntL, cv2.DIST_L2,0,0.01,0.01)
            #print('L angle = ' + str(np.arctan(vyL/vxL)) + 'R angle = ' + str(np.arctan(vyR/vxR)))
            #if (np.arctan(vyL/vxL) - np.arctan(vyR/vxR)) < 1.3: #less than 75 deg off
            slope_recipr = np.mean([(vxL/vyL),(vxR/vyR)])

        stereo_image_cropped_combined_gauss = cv2.line(stereo_image_cropped_combined_gauss,(int(225-225*(slope_recipr)),0),(int(225+225*(slope_recipr)),450),(0,255,0),2)
        cv2.imshow('with line', stereo_image_cropped_combined_gauss)
        

        
        
        
        
        #print(np.mean(stereo_image_filtered_cropped)*(150*150)/sum(stereo_mask_part_2))
#
#        for shift in range(100):
#            cv2.imshow('mask_overlay',frame_norm_L_masked + frame_norm_R_padded[100:-100,(100-shift):-(100+shift)])
#            if cv2.waitKey(70) & 0xFF == ord('q'):
#                break
#                
    else:
        # The next frame is not ready, so we try to read it again
        vid.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(100)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):   #cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
        break
    
    
    if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames, stop
        break

vid.release()
out.release()
#cv2.destroyAllWindows()


  