'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                                    Get Stimulus Response                                -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; from matplotlib import pyplot as plt; from scipy import linalg; import os; import sklearn; from sklearn.externals import joblib
from sklearn import mixture; from sklearn.model_selection import KFold; from hmmlearn import hmm; import warnings; warnings.filterwarnings('once')
from learning_funcs import filter_features, create_legend; import cv2; import pandas as pd

#%% -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
# Select model directory
file_location = 'C:\\Drive\\Video Analysis\\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'

concatenated_data_name_tag = 'analyze2' 
session_name_tag = ['normal_1_0', 'normal_1_1', 'normal_1_2']
p_value = .4

stimulus_frames = [10000,20000]
reaction_time_limit = 80 #frames (40 frame/sec)

session_video_folder = 'C:\\Drive\\Video Analysis\\data\\baseline_analysis\\27.02.2018\\205_1a\\'
session_video = session_video_folder + 'Chronic_Mantis_stim-default-996386-video-0.avi'



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Load and Prepare data                                        --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#find session
file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag 
session_name_tags = np.load(file_location_concatenated_data + '_session_name_tags.npy')

#load data     
if os.path.isfile(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy'):
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy')
else:
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity.npy')
    print('loaded non-flip-corrected data')
    
    
#for session in enumerate(session_name_tag):

print('')
file_location_data = file_location + data_folder + analysis_folder + session_name_tag[0] + '\\' + session_name_tag[0]
trials_in_session_index = np.zeros(position_orientation_velocity.shape[0]).astype(bool)

for name_tag in enumerate(session_name_tag):
    trials_in_session_index = trials_in_session_index + (position_orientation_velocity[:,7] == find(session_name_tags==name_tag[1]))

#print('session ' + str(session[0]+1) + ': ' + str(name_tag[0]+1) + ' videos found')
position_orientation_velocity_session = position_orientation_velocity[trials_in_session_index,:]

#find trials in bounds
out_of_bounds = position_orientation_velocity_session[:,1]

#find trials just after coming back in bounds
disruptions = np.ones(len(out_of_bounds)).astype(bool)
disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])
disruptions = disruptions[out_of_bounds == 0]

#get frame numbers of the trials to analyze
frames_all = np.arange(position_orientation_velocity_session.shape[0])
frames = frames_all[out_of_bounds==0]
ms_multiplier = 1000 / 40
    
    
#reaction time


#get distribution of angles toward shelter
if os.path.isfile(file_location_data + '_centre_shelter.npy'): #find shelter
    centre_shelter = np.load(file_location_data + '_centre_shelter.npy')
    shelter = centre_shelter[1]
else: #select centre of arena and shelter
    arena_vid = cv2.VideoCapture(file_location_data + '_normalized_video.avi')
    ret, frame = arena_vid.read(); print('select centre of arena')
    centre_roi = cv2.selectROI(frame[:,:,1])
    centre = [centre_roi[0]+centre_roi[2]/2, centre_roi[1]+centre_roi[3]/2]
    print('select centre of shelter'); shelter_roi = cv2.selectROI(frame[:,:,1]); print('thank you') #report centre of sheltre
    shelter = [shelter_roi[0]+shelter_roi[2]/2, shelter_roi[1]+shelter_roi[3]/2]
    np.save(file_location_data + '_centre_shelter.npy',[centre, shelter])    

#head direction relative to video frame (up is +90 deg)
head_direction = position_orientation_velocity_session[out_of_bounds==0,4] 
position = position_orientation_velocity_session[out_of_bounds==0,5:7]
shelter_direction = np.angle((shelter[0] - position[:,0]) + (position[:,1] - shelter[1])*1j,deg=True)

x = np.arange(0,180,1)
L = 1
k = .2
x0 = 10


def double_edged_sigmoid(x,L,k,x0,double=True):
    
    if double:
        x1 = 180-(x0/2)
        y = (L / (1 + np.exp(-k*(x-x0)))) * (L / (1 + np.exp(-k*(x1-x))))
    else:
        y = (L / (1 + np.exp(-k*(x-x0))))
#    plt.figure()
#    plt.scatter(x,y)
    return y


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
assert np.percentile(turn_toward_shelter,(1-p_value)*100) < 30
turn_toward_shelter[turn_toward_shelter > 30] = 30 
turn_toward_shelter[turn_toward_shelter < -30] = -30 

#make array length of video
turn_toward_shelter_full = np.ones(len(frames_all))*np.nan
head_direction_rel_to_shelter_full = np.ones(len(frames_all))*np.nan
turn_toward_shelter_full[frames] = turn_toward_shelter
head_direction_rel_to_shelter_full[frames] = head_direction_rel_to_shelter

mouse_vid = cv2.VideoCapture(session_video)

mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,800)

frame_rate = 10
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width = int(mouse_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(mouse_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
video = cv2.VideoWriter(session_video_folder + 'show_angles.avi',fourcc , frame_rate, (width,height)) 

while True:
    ret, frame = mouse_vid.read()
    if ret:
        frame_num = int(mouse_vid.get(cv2.CAP_PROP_POS_FRAMES))       
        
        cv2.putText(frame,str(frame_num),(50,1000),0,1,255)
        cv2.putText(frame,'HD: ' + str(int(head_direction_rel_to_shelter_show[frame_num-1])),(200,100),0,1,255)
        cv2.putText(frame,'turn: ' + str(int(turn_toward_shelter_full[frame_num-1])),(200,150),0,1,255)
        
        cv2.imshow('behaviour',frame)
        video.write(frame)
        
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break

video.release()
###get null distribution
reaction_time_limit = 80
angle_bins = np.arange(0,190,10)
significant_turn = np.zeros((reaction_time_limit,18))

for i in range(reaction_time_limit):
    turn_toward_shelter = np.zeros(len(head_direction_rel_to_shelter))
    last_head_direction = head_direction_rel_to_shelter[:-(i+1)]
    current_head_direction = head_direction_rel_to_shelter[(i+1):]
    turn_toward_shelter[(i+1):] = last_head_direction - current_head_direction
    
    #clip turn_toward_shelter to a reasonable value
    turn_toward_shelter[disruptions] == 0
    turn_toward_shelter[turn_toward_shelter > 30*(i+1)] = 30
    turn_toward_shelter[turn_toward_shelter < -30*(i+1)] = -30 

    turn_toward_shelter = turn_toward_shelter*double_edged_sigmoid(head_direction_rel_to_shelter,L,k,x0,double=True)
    for b in range(len(angle_bins)-1):
        angle_ind = np.zeros(len(turn_toward_shelter)).astype(bool)
        angle_ind[(i+1):] = (head_direction_rel_to_shelter[:-(i+1)]<angle_bins[b+1]) * (head_direction_rel_to_shelter[:-(i+1)]>=angle_bins[b])
        turn_toward_shelter_angle_bin = turn_toward_shelter[angle_ind]
        
        significant_turn[i,b] = np.percentile(turn_toward_shelter_angle_bin,(1-p_value)*100)

for stimulus_presentation in enumerate(stimulus_frames):
    for i in range(reaction_time_limit):
        turn_toward_shelter_full[stimulus_presentation[1]]
#








#speed toward shelter null distribution

#reaction time to run

#(or no response within 5 sec)