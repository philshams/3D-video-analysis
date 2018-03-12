# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:54:54 2018

@author: SWC
"""
import cv2
import os

file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'

file_loc = file_loc + date + mouse_session
#file_loc = ''

frame_rate = 20
display_frame_rate = 20
write_images = False
do_not_overwrite = True

#%%

# select videos to save 
fourcc = cv2.VideoWriter_fourcc(*'MJPG') #MJP2 /MJPG works; try MJP2 or LAGS or 'Y16 '; want uncompressed!!
save_file_loc = file_loc + 'Dario_snip_'

if write_images == True:

    video_file = save_file_loc + 'normal_video.avi'
    if os.path.isfile(video_file) and do_not_overwrite:
        raise Exception('File already exists') 
    normal_video = cv2.VideoWriter(video_file,fourcc , frame_rate, (width,height), False) 

    video_file = save_file_loc + 'normalized_video.avi'
    if os.path.isfile(video_file) and do_not_overwrite:
        raise Exception('File already exists')         
    normalized_video = cv2.VideoWriter(video_file,fourcc , frame_rate, (width,height), False) 

    video_file = save_file_loc + 'cropped_mouse.avi'
    if os.path.isfile(video_file) and do_not_overwrite:
        raise Exception('File already exists')         
    cropped_mouse = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450), False) 

    video_fileL = save_file_loc + 'stereo_input_L.avi'
    video_fileR = save_file_loc + 'stereo_input_R.avi'
    if (os.path.isfile(video_fileL) or os.path.isfile(video_fileR)) and do_not_overwrite:
        raise Exception('File already exists') 
    stereo_input_L = cv2.VideoWriter(video_fileL,fourcc , frame_rate, (350,200), False) 
    stereo_input_R = cv2.VideoWriter(video_fileR,fourcc , frame_rate, (350,200), False) 

    video_file = save_file_loc + '3D_single_pass.avi'
    if os.path.isfile(video_file) and do_not_overwrite:
        raise Exception('File already exists') 
    threeD_individual = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450), False )

    video_file = save_file_loc + '3D_combined.avi'
    if os.path.isfile(video_file) and do_not_overwrite:
        raise Exception('File already exists')         
    threeD_combined = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450), True) 

    video_file = save_file_loc + '3D_smooth.avi'
    if os.path.isfile(video_file) and do_not_overwrite:
        raise Exception('File already exists') 
    threeD_smooth = cv2.VideoWriter(video_file,fourcc , frame_rate, (450,450), True) 


#%%

vid1 = cv2.VideoCapture(file_loc + 'normal_video.avi')  
vid2 = cv2.VideoCapture(file_loc + 'normalized_video.avi')  
vid3 = cv2.VideoCapture(file_loc + 'cropped_mouse.avi')  
vid4 = cv2.VideoCapture(file_loc + 'stereo_input_L.avi')
vid5 = cv2.VideoCapture(file_loc + 'stereo_input_R.avi')
vid6 = cv2.VideoCapture(file_loc + '3D_single_pass.avi')
vid7 = cv2.VideoCapture(file_loc + '3D_combined.avi')
vid8 = cv2.VideoCapture(file_loc + '3D_smooth.avi')

start_frame = 2000
end_frame = 10000
vid1.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid2.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid3.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid4.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid5.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid6.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid7.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
vid8.set(cv2.CAP_PROP_POS_FRAMES,start_frame)


frame_no = start_frame

while True:
    ret, frame1 = vid1.read() # get the frame
    ret, frame2 = vid2.read() # get the frame
    ret, frame3 = vid3.read() # get the frame
    ret, frame4 = vid4.read() # get the frame
    ret, frame5 = vid5.read() # get the frame
    ret, frame6 = vid6.read() # get the frame
    ret, frame7 = vid7.read() # get the frame
    ret, frame8 = vid8.read() # get the frame
    if ret: 
        
        print(frame_no)
        
        cv2.imshow('normal',frame1)
        cv2.imshow('normalized',frame2)
        cv2.imshow('cropped',frame3)
        cv2.imshow('stereo input L',frame4)
        cv2.imshow('stereo input R',frame5)
        cv2.imshow('3D single pass',frame6)
        cv2.imshow('3D combined',frame7)
        cv2.imshow('3D smooth',frame8)
        
        frame_no +=1
        
        if write_images:
            normal_video.write(frame2)   
            normalized_video.write(frame2)
            cropped_mouse.write(frame3)
            stereo_input_L.write(frame4) 
            stereo_input_R.write(frame5) 
            threeD_individual.write(frame6)
            threeD_combined.write(frame7)
            threeD_smooth.write(frame8) 

        if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
            break
        if vid1.get(cv2.CAP_PROP_POS_FRAMES) > min(end_frame,vid1.get(cv2.CAP_PROP_FRAME_COUNT)):
            break 
        
vid1.release()
vid2.release()
vid3.release()
vid4.release()
vid5.release()
vid6.release()
vid7.release()
vid8.release()

if write_images:
    normal_video.release()   
    normalized_video.release()
    cropped_mouse.release()
    stereo_input_L.release() 
    stereo_input_R.release() 
    threeD_individual.release()
    threeD_combined.release()
    threeD_smooth.release()


 