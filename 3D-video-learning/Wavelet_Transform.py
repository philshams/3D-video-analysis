# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:39:50 2018

@author: SWC
"""
import numpy as np
import pywt
import cv2


#Receive data video
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
file_loc = file_loc + date + mouse_session

save_vid_name = 'analyze_3_5'
file_loc = file_loc + save_vid_name

vid = cv2.VideoCapture(file_loc + '_data.avi')   

frame_rate = 1000
stop_frame = 17000
level = 5 # how many different spatial scales to use
discard_scale = 4 #4 discards 4/5; 6 keeps all


#%%

num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
wavelet_mouse = np.zeros((39,39,stop_frame)).astype(np.float16)


# for each frame (resize?) and perform wavelet decomposition
while True:
    ret, frame = vid.read() # get the frame

    if ret: 
        frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
        frame = frame[:,:,0]
        #cv2.imshow('normal',frame)
        
        #frame = np.ones((6,6))*[1,2,100,4,20,200]
        coeffs_lowpass = [[],[],[],[],[],[]]
        coeffs = pywt.wavedec2(frame, wavelet='db1',level = level)
        
        for i in range(level+1):
            if i < discard_scale:
                coeffs_lowpass[i] = coeffs[i]
            else:
                coeffs_lowpass[i] = [None,None,None]

        wavelet_recon = pywt.waverec2(coeffs_lowpass, wavelet='db1').astype(np.uint8)
        #cv2.imshow('wavelet reconstruction2',wavelet_recon)
        
        coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs[0:discard_scale])
#        features = np.hstack([ib.ravel() for sublist in coeffs[0:discard_scale] for ib in sublist])
        wavelet_mouse[:,:,frame_num-1] = coeff_array
        
        
        if (frame_num)%500==0:
            print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')        
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break
        if vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame: #vid.get(cv2.CAP_PROP_FRAME_COUNT):
            break 
        
    else:
        print('broken...')
        
vid.release()
np.save(file_loc + 'wavelet_mouse_3_5',wavelet_mouse)
np.save(file_loc + 'wavelet_slices_mouse_3_5',coeff_slices)


    #verify wavelet decomposition success with k dim