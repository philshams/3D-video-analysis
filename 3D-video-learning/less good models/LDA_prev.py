'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; import sklearn.decomposition; import os; from shutil import copyfile
from learning_funcs import reconstruct_from_wavelet; from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis; import glob

''' -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''

# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"
data_library_name_tag = 'analyze'

do_not_overwrite = True


# ---------------------------
# Select analysis parameters
# ---------------------------
session_to_examine = 0
display_frame_rate = 100

num_LDCs = 4
flip_threshold = .81
feature_relevance_threshold = 0.1; modify_relevant_features_from_previous_runs = True
every_other = 2 #if too much data to store in memory, downsample























''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load wavelet transformed data                    -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''


# ---------------------------
# Set up data library folder
# ---------------------------
folder_location_data_library = save_folder_location + data_library_name_tag + '\\'
file_location_data_library = folder_location_data_library + data_library_name_tag
print("saving to " + folder_location_data_library)


# ---------------------------
# Set up data library folder
# ---------------------------
session_name_tags = np.load(file_location_data_library + '_session_name_tags.npy')
print(session_name_tags)
session_folders = np.concatenate((np.array(session_name_tags),np.array(session_name_tags)))


# ------------------------------------------------------
# calculate relevant features in the wavelet transform
# ------------------------------------------------------
if modify_relevant_features_from_previous_runs or not os.path.isfile(file_location_data_library + '_wavelet_relevant_ind_LDA.npy'):
    print('calculating relevant features...')
    feature_used_sum_together = np.zeros(39*39); feature_used_std_together = np.zeros(39*39)

    # loop over each video, adding together wavelet-arrays to find which values are actually used
    v = 0  
    for session in enumerate(session_name_tags):
        file_locations_saved_data = glob.glob(save_folder_location + session[1] + '\\' + '*_wavelet.npy')
        print(session[1])
        print('found ' + str(len(file_locations_saved_data)) + ' data files')
        for wavelet_video in enumerate(file_locations_saved_data):    
            wavelet_array = np.load(wavelet_video[1]).astype(np.float16)
            wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))

            # Use only features that have non-zero variation for the LDA            
            feature_used_std = np.std(wavelet_array.T,axis=0, dtype=np.float64)
            feature_used_sum_together = feature_used_sum_together + feature_used_std
            feature_used_std_thresholded = feature_used_std > feature_relevance_threshold
            feature_used_std_together = feature_used_std_together + feature_used_std_thresholded
            v += 1
    feature_used_sum_together = feature_used_sum_together / v            
    feature_used_std_together = feature_used_std_together / v # now represent percent of data files with non-zero variation
    relevant_wavelet_features = (feature_used_std_together > feature_relevance_threshold) *  \
                                   (feature_used_sum_together > feature_relevance_threshold) #change to zero to keep more features
    
    # also save the index of each of these features
    relevant_wavelet_features = np.where(relevant_wavelet_features)[0]
    np.save(file_location_data_library + '_wavelet_relevant_ind_LDA.npy', relevant_wavelet_features)
else:
    relevant_wavelet_features = np.load(file_location_data_library + '_wavelet_relevant_ind_LDA.npy')
coeff_slices = np.load(save_folder_location + 'wavelet_slices.npy')

print(str(len(relevant_wavelet_features)) + ' relevant features retained from wavelet transform')

 

''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Gather Data for LDA                                  -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------------------------------------------------
# Create huge array of wavelet features from all videos, both straight and upside-down
# ------------------------------------------------------------------------------------
print('preparing features...')
wavelet_array_relevant_features_up = np.zeros((1,len(relevant_wavelet_features)))
wavelet_array_relevant_features_down = np.zeros((1,len(relevant_wavelet_features)))
session_index = []

#for each session, add too the array the up and down wavelet features
for session in enumerate(session_name_tags):
    file_locations_saved_data = glob.glob(save_folder_location + session[1] + '\\' + '*wavelet.npy')
    wavelet_array_session_up = np.zeros((1,len(relevant_wavelet_features)))
    wavelet_array_session_down = np.zeros((1,len(relevant_wavelet_features)))
    #do so for every video
    for wavelet_video in enumerate(file_locations_saved_data): 
        wavelet_array = np.load(wavelet_video[1]).astype(np.float64)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        if wavelet_video[1].find('upside') > 0:  # add this video's array to huge array; downsample if necessary
            wavelet_array_session_down = np.concatenate((wavelet_array_session_down, wavelet_array[relevant_wavelet_features, :].T))
        else:
            wavelet_array_session_up = np.concatenate((wavelet_array_session_up, wavelet_array[relevant_wavelet_features, :].T))
    wavelet_array_relevant_features_down = np.concatenate((wavelet_array_relevant_features_down, wavelet_array_session_down[1::every_other,:]))
    wavelet_array_relevant_features_up = np.concatenate((wavelet_array_relevant_features_up, wavelet_array_session_up[1::every_other,:]))
    #create session index for displaying particular sessions below
    session_index = np.concatenate((session_index, np.zeros(wavelet_array[relevant_wavelet_features, ::every_other].shape[1]) + session[0]), axis = 0)
wavelet_array_relevant_features_up = wavelet_array_relevant_features_up[1:,:]
wavelet_array_relevant_features_down = wavelet_array_relevant_features_down[1:,:]



# ----------------------------------
# Add forward velocity as a feature
# ----------------------------------
print('appending velocity and rescaling...')
position_orientation_velocity = np.load(file_location_data_library + '_position_orientation_velocity.npy')
out_of_bounds = position_orientation_velocity[:,1]
disruptions = np.ones(len(out_of_bounds)).astype(bool)
disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])

# downsample data if necessary
if every_other > 1:
    position_orientation_velocity = position_orientation_velocity[out_of_bounds==0,:]
    disruptions = disruptions[out_of_bounds==0]
    trials_to_analyze_index = np.array([]).astype(int)
    for session in enumerate(session_name_tags):
        file_locations_saved_data = glob.glob(save_folder_location + session[1] + '\\' + '*_data.avi')      
        trials_to_analyze_index_cur = np.where(position_orientation_velocity[:,7]==session[0])[0][::every_other]
        trials_to_analyze_index = np.append(trials_to_analyze_index, trials_to_analyze_index_cur)
else: # use all trials where the mouse was in bounds
    trials_to_analyze_index = out_of_bounds==0

velocity = position_orientation_velocity[trials_to_analyze_index,2:3] #velocity toward head direction
disruptions = disruptions[trials_to_analyze_index]
velocity[disruptions==1,0] = 0 #remove spurious velocities
mean_vel = np.mean(velocity[:]); std_vel = np.std(velocity[:])
greater_than_median_vel_ind = np.squeeze(velocity > np.median(velocity) + std_vel)
greater_than_sub_median_vel_ind = np.squeeze(velocity >= np.median(velocity) - std_vel)
less_than_super_median_vel_ind = np.squeeze(velocity <= np.median(velocity) + std_vel)
print('only flip frames below ' + str(np.median(velocity) + std_vel) + ' pixels/frame')
velocity[velocity[:,0]-mean_vel > 6*std_vel,0] = mean_vel + 6*std_vel
velocity[velocity[:,0]-mean_vel < -6*std_vel,0] = mean_vel - 6*std_vel #saturate velocity below spurious level

# -------------
# rescale data
# -------------
velocity_scaler = sklearn.preprocessing.RobustScaler()
velocity_scaler.fit(velocity)
velocity = velocity_scaler.transform(velocity)

up_means = np.mean(wavelet_array_relevant_features_up,axis=0)
up_stds = np.std(wavelet_array_relevant_features_up,axis=0)
if sum(up_stds==0) or sum(isnan(up_stds==0)):
    raise Exception('some so-called relevant features with no variation found -- increase threshold')    
wavelet_array_relevant_features_up = ((wavelet_array_relevant_features_up - up_means) / up_stds)
wavelet_array_relevant_features_down = ((wavelet_array_relevant_features_down - up_means) / up_stds)

# --------------------------------------
# concatenate features and create labels
# --------------------------------------
features_up = np.concatenate((wavelet_array_relevant_features_up,velocity.astype(np.float16)),axis=1)
features_down = np.concatenate((wavelet_array_relevant_features_down,-1*velocity.astype(np.float16)),axis=1)
data_for_classifier = np.concatenate((features_up, features_down), axis = 0)
labels_up = np.ones(features_up.shape[0])
labels_down = -1 * np.ones(features_down.shape[0])
labels_for_classifier = np.concatenate((labels_up,labels_down), axis = 0 )
print('')

        
''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Get LDCs                                        -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''
    
# ------------------------------------------
# Generate the LDCs for the wavelet features
# ------------------------------------------
for i in range(2): #do an additional round, after switching the labels for the definitely-upside-down frames
    print('')
    print('fitting lda model...')
    lda = LinearDiscriminantAnalysis(solver='lsqr', n_components=num_LDCs, store_covariance=False, tol=0.0001)
    lda.fit(data_for_classifier, labels_for_classifier) # input: (samples, features)
    accuracy = lda.score(features_up, labels_up)

    if i==0: 
        print('round 1/2 performance:')
        flip_up = (lda.predict_proba(features_up)[:,0] > .99) * less_than_super_median_vel_ind
        flip_down = (lda.predict_proba(features_down)[:,1] > .99)*greater_than_sub_median_vel_ind
        labels_for_classifier[:len(flip_up)] = labels_up - 2* flip_up
        labels_for_classifier[-len(flip_down):] = labels_down + 2*flip_down
    else:
        print('round 2/2 performance:')
    print('accuracy of ' + str(accuracy))

    # --------------------------------------------
    # Display the frames classified as upside-down
    # --------------------------------------------
    #calculate probabilities/predictions for a particular session
    print('examining session: ' + session_name_tags[session_to_examine])
    features_lda = features_up[session_index==session_to_examine,:]
    predicted_prob_up = lda.predict_proba(features_up[session_index==session_to_examine,:])
    predicted_state_flip = (predicted_prob_up[:,0] > flip_threshold) * less_than_super_median_vel_ind[session_index == session_to_examine]
    flip_ind = np.where(predicted_state_flip==True)[0]
    keep_ind = np.where(predicted_state_flip==False)[0]
    print(str(len(flip_ind)) + ' flips out of ' + str(len(predicted_state_flip)) + ' frames')
    
    # Open up the selected data wavelet array
    file_location_session = save_folder_location + session_name_tags[session_to_examine] + '\\' + session_name_tags[session_to_examine]
    wavelet_array_session = np.load(file_location_session + '_wavelet.npy')[:,:,::every_other]
    
    #show in one video the unflipped frames, and in another, those that the model would flip
    level = 5; discard_scale = 4  # these must be parameters taken from original wavelet transform
    for j in range(len(flip_ind)):
        #reconstruct images from wavelet transform
        wavelet_up = wavelet_array_session[:,:,keep_ind[j]]; wavelet_down = wavelet_array_session[:,:,flip_ind[j]]
        reconstruction_from_wavelet_up  = reconstruct_from_wavelet(wavelet_up,coeff_slices, level, discard_scale)
        reconstruction_from_wavelet_down  = reconstruct_from_wavelet(wavelet_down,coeff_slices, level, discard_scale)
        cv2.imshow('right-side up',reconstruction_from_wavelet_up.astype(np.uint8))
        cv2.imshow('up-side down',reconstruction_from_wavelet_down.astype(np.uint8))
        i+=1
        if (cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q')) or i >= len(flip_ind) - 1:
            break
        if i%500==0:
            print(str(j) + ' out of ' + str(len(flip_ind)) + ' frames complete')


# ---------------
# Save the model
# ---------------
save_file_model = file_location_data_library + '_lda'
if os.path.isfile(save_file_model) and do_not_overwrite:
    raise Exception('File already exists')
joblib.dump(lda, save_file_model)
np.save(save_file_model + '_scaling', np.array([[mean_vel],[std_vel],up_means,up_stds]))
joblib.dump(velocity_scaler, save_file_model + '_velocity_scaling')
np.save(file_location_data_library + '_session_index',session_index)

    
    
    
    

