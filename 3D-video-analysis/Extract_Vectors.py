'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                                    View / Update Mouse's Vector with Respect to Shelter                          -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; from matplotlib import pyplot as plt; from scipy import linalg; import os; import sklearn; from sklearn.externals import joblib
from sklearn import mixture; from sklearn.model_selection import KFold; from hmmlearn import hmm; import warnings; warnings.filterwarnings('once')
from analysis_funcs import filter_features, create_legend, double_edged_sigmoid; import cv2; import pandas as pd

#%% -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"

concatenated_data_name_tag = 'streamlined' 
session_name_tag = 'loom_1'

session_video_folder = 'C:\\Drive\\Video Analysis\\data\\baseline_analysis\\27.02.2018\\205_1a\\'
session_video = session_video_folder + 'Chronic_Mantis_stim-default-996386-video-0.avi'
video_num = 0

start_frame = 700
undistort = True
frame_rate = 10

































#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Load and Prepare data                                        --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------
# load data    
# ------------ 
file_location_data_cur = save_folder_location + session_name_tag + '\\' + session_name_tag
if os.path.isfile(file_location_data_cur + '_position_orientation_velocity_corrected.npy'):
    position_orientation_velocity = np.load(file_location_data_cur + '_position_orientation_velocity_corrected.npy')
else:
    position_orientation_velocity = np.load(file_location_data_cur + '_position_orientation_velocity.npy')
    print('loaded non-flip-corrected data')
   
    
#find trials in bounds
out_of_bounds = position_orientation_velocity[:,1]

#find trials just after coming back in bounds
disruptions = np.ones(len(out_of_bounds)).astype(bool)
disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])
disruptions = disruptions[out_of_bounds == 0]

#get frame numbers of the trials to analyze
frames_all = np.arange(position_orientation_velocity.shape[0])
frames = frames_all[out_of_bounds==0]
      

    
#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Analyze  data                                        --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

    
# -------------------------------------------
# get distribution of angles toward shelter
# -------------------------------------------
# get shelter position
if os.path.isfile(file_location_data_cur + '_shelter_roi.npy'):
    shelter = np.load(file_location_data_cur + '_shelter_roi.npy')
else:
    raise Exception('please go back create arena and shelter rois from preprocessing script')
shelter_center = [shelter[0] + shelter[2] / 2, shelter[1] + shelter[3] / 2]


#head direction relative to video frame (up is +90 deg)
head_direction = position_orientation_velocity[out_of_bounds==0,4:5] 
position = position_orientation_velocity[out_of_bounds==0,5:7]
shelter_direction = np.angle((shelter_center[0] - position[:,0]) + (position[:,1] - shelter_center[1])*1j,deg=True)
distance_from_shelter = np.sqrt((shelter_center[0] - position[:,0])**2 + (position[:,1] - shelter_center[1])**2)

# get absolute head turn
angular_speed_for_clipping = np.zeros(head_direction.shape[0])
last_head_direction = head_direction[:-1, :]; current_head_direction = head_direction[1:, :]
angular_speed = np.min(np.concatenate((abs(current_head_direction - last_head_direction),
                                   abs(360 - abs(current_head_direction - last_head_direction))), axis=1), axis=1)

# assume that very large turns in a single frame are spurious
angular_speed[angular_speed > 180] = abs(360 - angular_speed[angular_speed > 180])
angular_speed[disruptions[1:]] = 0
angular_speed_for_clipping[1:] = angular_speed
head_direction = head_direction[:,0]


# set parameters for the sigmoid non-linearity
L = 1
k = .2
x0 = 10

#head direction relative to shelter
head_direction_rel_to_shelter = abs(head_direction - shelter_direction) #left and right counted as equal
head_direction_rel_to_shelter[head_direction_rel_to_shelter>180] = abs(360 - head_direction_rel_to_shelter[head_direction_rel_to_shelter>180])
head_direction_rel_to_shelter_show = head_direction_rel_to_shelter*double_edged_sigmoid(head_direction_rel_to_shelter,L,k,x0,double=False)

#change in head direction relative to shelter
turn_toward_shelter = np.zeros(len(head_direction_rel_to_shelter))
last_head_direction = head_direction_rel_to_shelter[:-1]
current_head_direction = head_direction_rel_to_shelter[1:]
turn_toward_shelter[1:] = last_head_direction - current_head_direction
turn_toward_shelter[disruptions] == 0
turn_toward_shelter = turn_toward_shelter*double_edged_sigmoid(head_direction_rel_to_shelter,L,k,x0,double=True)

#clip turn_toward_shelter to a reasonable value
turn_toward_shelter[angular_speed_for_clipping > 90] = 0
turn_toward_shelter[turn_toward_shelter > 90] = 180 - turn_toward_shelter[turn_toward_shelter > 90]
turn_toward_shelter[turn_toward_shelter < -90] = -180 - turn_toward_shelter[turn_toward_shelter < -90]
turn_toward_shelter[turn_toward_shelter > 30] = 30 
turn_toward_shelter[turn_toward_shelter < -30] = -30 

#make array length of video
turn_toward_shelter_full = np.ones(len(frames_all))*np.nan; head_direction_rel_to_shelter_full = np.ones(len(frames_all))*np.nan
distance_from_shelter_full = np.ones(len(frames_all))*np.nan; position_full = np.ones((len(frames_all),2))*np.nan
head_direction_full = np.ones(len(frames_all))*np.nan
turn_toward_shelter_full[frames] = turn_toward_shelter; head_direction_rel_to_shelter_full[frames] = head_direction_rel_to_shelter
distance_from_shelter_full[frames] = distance_from_shelter; head_direction_full[frames] = head_direction
position_full[out_of_bounds==1,:] = 0; position_full[frames,:] = position
position_full[out_of_bounds==1,:] = shelter_center; position_full = position_full.astype(int)



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Show video with orientation vector                                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


mouse_vid = cv2.VideoCapture(session_video)
mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
width = int(mouse_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(mouse_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#video = cv2.VideoWriter(session_video_folder + 'show_angles.avi',fourcc , frame_rate, (width,height)) 


if undistort:
    maps = np.load(save_folder_location + "fisheye_maps.npy")
    map1 = maps[:, :, 0:2]
    map2 = maps[:, :, 2]



print('press q to quit')
print('press d to fast forward video')
print('press a to slow down video')
print('press s to go back 10 frames and reset frame rate')
print('press w to toggle on flip mode:')
print('go frame by frame via any key, and space bar to add frame to flip index')



flip_ind = []
additional_pause = 0
original_frame_rate = frame_rate

while True:
    ret, frame = mouse_vid.read()
    if ret:
        frame_num = int(mouse_vid.get(cv2.CAP_PROP_POS_FRAMES)) - 1   
        if undistort:
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        cv2.putText(frame,str(frame_num),(50,1000),0,1,255)
        cv2.putText(frame,'HD: ' + str(int(head_direction_rel_to_shelter_show[frame_num])),(200,100),0,1,255)
        cv2.putText(frame,'turn: ' + str(int(turn_toward_shelter_full[frame_num])),(200,150),0,1,255)
        if out_of_bounds[frame_num]==0:
            hd_cur = head_direction_full[frame_num]
            hd_shelter_cur = head_direction_rel_to_shelter_full[frame_num]
            red = (180 - hd_shelter_cur)*250 / 180
            green = (hd_shelter_cur)*250 / 180
            cv2.arrowedLine(frame, (position_full[frame_num,0], position_full[frame_num,1]),
                            (int(position_full[frame_num,0] + 30*np.cos(hd_cur * np.pi / 180)), 
                             int(position_full[frame_num,1] - 30*np.sin(hd_cur * np.pi / 180))), (0, green, red), thickness=2)
        elif out_of_bounds[frame_num]==1:
            cv2.circle(frame, (shelter_center[0], shelter_center[0]), 10, (255,0,0), -1)
            
        cv2.imshow('behaviour',frame)
#        video.write(frame)
        
        keystroke = cv2.waitKey(int(1000/frame_rate) + additional_pause)
        additional_pause = 0
        
        if keystroke == ord('q'):
            save_choice = input('save flip ind [y/n] ?')
            if save_choice == 'y':
                save_flip_ind = True
            else:
                save_flip_ind = False
            
            break
        elif keystroke == ord('d'):
            frame_rate = min(1000,frame_rate * 1.2)
            print(frame_rate)
        elif keystroke == ord('a'):
            frame_rate = max(1,frame_rate / 1.2)
            print(frame_rate)
        elif keystroke == ord('w'):
            if frame_rate > 0:
                frame_rate = -1             
            else:
                frame_rate = original_frame_rate
        elif keystroke == ord('s'):
            mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,frame_num - 10)
            additional_pause = 500
            frame_rate = original_frame_rate
        elif keystroke == 32: #space
            flip_ind.append(frame_num)
            print(frame_num)


#video.release()


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Show flipped frames                                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

if len(flip_ind) > 0:
    flip_counter = 0
    mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,flip_ind[flip_counter])
    while True:
        ret, frame = mouse_vid.read()
        if ret:
            flip_counter += 1
            frame_num = int(mouse_vid.get(cv2.CAP_PROP_POS_FRAMES)) - 1      
            if undistort:
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            cv2.putText(frame,str(frame_num),(50,1000),0,1,255)
            cv2.putText(frame,'HD: ' + str(int(head_direction_rel_to_shelter_show[frame_num])),(200,100),0,1,255)
            cv2.putText(frame,'turn: ' + str(int(turn_toward_shelter_full[frame_num])),(200,150),0,1,255)
            if out_of_bounds[frame_num]==0:
                hd_cur = head_direction_full[frame_num]
                hd_shelter_cur = head_direction_rel_to_shelter_full[frame_num]
                red = (180 - hd_shelter_cur)*250 / 180
                green = (hd_shelter_cur)*250 / 180
                cv2.arrowedLine(frame, (position_full[frame_num,0], position_full[frame_num,1]),
                                (int(position_full[frame_num,0] + 30*np.cos(hd_cur * np.pi / 180)), 
                                 int(position_full[frame_num,1] - 30*np.sin(hd_cur * np.pi / 180))), (0, green, red), thickness=2)
            elif out_of_bounds[frame_num]==1:
                cv2.circle(frame, (shelter_center[0], shelter_center[0]), 10, (255,0,0), -1)
                
            cv2.imshow('behaviour',frame)
            
            if flip_counter >= len(flip_ind):
                break
            
            keystroke = cv2.waitKey(int(1000/5))
            if keystroke == ord('q'):
                break

















