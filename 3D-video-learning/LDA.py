'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; import sklearn.decomposition; import os; from shutil import copyfile
from learning_funcs import reconstruct_from_wavelet; from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_location = 'C:\Drive\Video Analysis\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'
session_name_tags = ['normal_1_0','normal_1_1','normal_1_2']

''',
                    'normal_2_0','normal_2_1','normal_2_2',
                    'normal_3_0','normal_3_1',
                    'normal_4_0','normal_4_1','normal_4_2',
                    'normal_5_0','normal_5_1',
                    'normal_6_0','normal_6_1','normal_6_2',
                    'clicks_1_0','clicks_2_0','clicks_3_0',
                    'post_clicks_1_0','post_clicks_2_0','post_clicks_3_0']
'''

concatenated_data_name_tag = 'all'

# ---------------------------
# Select analysis parameters
# ---------------------------
fetch_extant_LDA = False #tweak parameters of an existing LDA
save_LDA = False

num_LDCs = 2 #1 or 2 should suffice
flip_threshold = .85 #.85 should suffice

do_not_overwrite = False
examine_LDA = True
session_to_examine = 'normal_1_1'
display_frame_rate = 10

# these must be parameters taken from original wavelet transform 
level = 5
discard_scale = 4



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load wavelet transformed data                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

session_name_tags_straight = session_name_tags
session_name_tags_upside_down = [x + '_upside_down' for x in session_name_tags]
session_name_tags = np.concatenate((np.array(session_name_tags),np.array(session_name_tags_upside_down)))
session_folders = np.concatenate((np.array(session_name_tags_straight),np.array(session_name_tags_straight)))

if not fetch_extant_LDA:
    # ------------------------------------------
    # Load wavelet-transformed data from file
    # ------------------------------------------
    print('calculating relevant features...')

    file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag
    feature_used_sum_together = np.zeros(39*39)
    for v in range(len(session_name_tags)):
        file_location_wavelet = file_location + data_folder + analysis_folder + session_folders[v] + '\\' + session_name_tags[v]
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
        
        wavelet_array = np.load(wavelet_file)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        coeff_slices = np.load(wavelet_slices_file)
        
        # ---------------------------------------------------
        # Use only features that have non-zero values for PCA
        # ---------------------------------------------------
        feature_used = wavelet_array!=0
        feature_used_sum = np.mean(feature_used,axis=1)
        feature_used_sum_together = feature_used_sum_together + abs(feature_used_sum)
    
    relevant_features = feature_used_sum_together > 0.01 #change to zero to keep more features
    
    # also save the index of each of these features
    relevant_ind = find(relevant_features)
    np.save(file_location_concatenated_data + '_wavelet_relevant_ind_LDA.npy', relevant_ind)
else:
    relevant_ind = np.load(file_location_concatenated_data + '_wavelet_relevant_ind_LDA.npy')
    
print(str(sum(relevant_features)) + ' relevant features retained from wavelet transform')
print('')
    
    
    
#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Gather Data for LDA                                  -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

if not fetch_extant_LDA:
    print('preparing features...')
    wavelet_array_relevant_features_up = np.zeros((1,len(relevant_ind)))
    for v in range(len(session_name_tags_straight)):
        file_location_wavelet = file_location + data_folder + analysis_folder + session_folders[v] + '\\' + session_name_tags_straight[v]
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
        
        wavelet_array = np.load(wavelet_file)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        
        # create resulting output features and mean values to be used in PCA
        wavelet_array_relevant_features_up = np.concatenate((wavelet_array_relevant_features_up, wavelet_array[relevant_features,:].T))
    wavelet_array_relevant_features_up = wavelet_array_relevant_features_up[1:,:]
    
    print('preparing upside-down features...') 
    wavelet_array_relevant_features_down = np.zeros((1,len(relevant_ind)))
    for v in range(len(session_name_tags_upside_down)):
        file_location_wavelet = file_location + data_folder + analysis_folder + session_folders[v] + '\\' + session_name_tags_upside_down[v]
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
        
        wavelet_array = np.load(wavelet_file)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        
        # create resulting output features and mean values to be used in PCA
        wavelet_array_relevant_features_down = np.concatenate((wavelet_array_relevant_features_down, wavelet_array[relevant_features,:].T))
    wavelet_array_relevant_features_down = wavelet_array_relevant_features_down[1:,:]
    
    print('appending velocity and rescaling...')
    #append velocity
    velocity = np.load(file_location_concatenated_data + '_velocity.npy')
    disruptions = np.load(file_location_concatenated_data + '_disruption.npy')
    
    velocity = velocity[:,0:1]
    velocity[disruptions==1,0] = 0 #remove spurious velocities
    mean_vel = np.mean(velocity[:])
    std_vel = np.std(velocity[:])
    velocity[abs(velocity[:,0]-mean_vel) > 6*std_vel,0] = 0
    
    #rescale data
    velocity = sklearn.preprocessing.robust_scale(velocity)
    up_means = np.mean( wavelet_array_relevant_features_up,axis=0)
    up_stds = np.std( wavelet_array_relevant_features_up,axis=0)
    wavelet_array_relevant_features_up = (wavelet_array_relevant_features_up - up_means) / up_stds
    wavelet_array_relevant_features_down = (wavelet_array_relevant_features_down - up_means) / up_stds
    
    
    # append data and create labels
    features_up = np.concatenate((wavelet_array_relevant_features_up,velocity[0:wavelet_array_relevant_features_up.shape[0]]),axis=1)
    features_down = np.concatenate((wavelet_array_relevant_features_down,-1*velocity[0:wavelet_array_relevant_features_up.shape[0]]),axis=1)
    data_for_classifier = np.concatenate((features_up, features_down), axis = 0)
    labels_up = np.ones(features_up.shape[0])
    labels_down = -1 * np.ones(features_down.shape[0])
    labels_for_classifier = np.concatenate((labels_up,labels_down), axis = 0 )
    print('')

        
#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Get LDCs                                        -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
    
# ------------------------------------------
# Generate the LDCs for the wavelet features
# ------------------------------------------
if not fetch_extant_LDA:
    print('fitting lda model...')
    lda = LinearDiscriminantAnalysis(solver='lsqr', n_components=num_LDCs, store_covariance=False, tol=0.0001)
    lda.fit(data_for_classifier, labels_for_classifier) # input: (samples, features)
else: 
    lda = joblib.load(file_location_concatenated_data + '_lda')
    
accuracy = lda.score(features_up, labels_up)
print('accuracy of ' + str(accuracy))
    
    

predicted_prob_up = lda.predict_proba(features_up)
predicted_state_flip = predicted_prob_up[:,0] > flip_threshold

flip_ind = find(predicted_state_flip==True)
keep_ind = find(predicted_state_flip==False)
print(str(len(flip_ind)) + ' flips')


# --------------------------------------------
# Display the frames classified as upside-down
# --------------------------------------------

# Open up the selected data video file, and make a copy
file_location_movie = file_location + data_folder + analysis_folder + session_to_examine + '\\' + session_to_examine
vid_up = cv2.VideoCapture(file_location_movie + '_data.avi')

if not os.path.isfile(file_location_movie + '_data_copy.avi'):
    copyfile(file_location_movie + '_data.avi', file_location_movie + '_data_copy.avi')
    
vid_down = cv2.VideoCapture(file_location_movie + '_data_copy.avi')  

i = 0 #show in one video the unflipped frames, and in another, those that the model would flip
while True:
    vid_up.set(cv2.CAP_PROP_POS_FRAMES,keep_ind[i])
    vid_down.set(cv2.CAP_PROP_POS_FRAMES,flip_ind[i])    

    ret1, frame_up = vid_up.read() # get the frame
    ret2, frame_down = vid_down.read() # get the frame

    if ret1 and ret2: 
        
        cv2.imshow('right-side up',frame_up)
        cv2.imshow('up-side down',frame_down)
        i+=1
        
        if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
            break
        
        if i%500==0:
            print(str(i) + ' out of ' + str(len(flip_ind)) + ' frames complete')
            
        if i >= len(flip_ind) - 1:
            break 
    else:
        print('Problem with movie playback')
        cv2.waitKey(100)
        break
        
vid_up.release()
vid_down.release()


# ---------------
# Save the model
# --------------- 
if save_LDA:    
    save_file_model = file_location_concatenated_data + '_lda'
    if os.path.isfile(save_file_model) and do_not_overwrite:
        raise Exception('File already exists') 
    joblib.dump(lda, save_file_model)
    

    
    
    
    

