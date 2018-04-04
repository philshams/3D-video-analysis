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
concatenated_data_name_tag = 'analyze3'


# ---------------------------
# Select analysis parameters
# ---------------------------
save_LDA = True
fetch_extant_LDA = False #tweak parameters or view performance of an existing LDA

do_not_overwrite = False
examine_LDA = True
session_to_examine = 0
display_frame_rate = 30



# these must be parameters taken from original wavelet transform 
level = 5
discard_scale = 4
num_LDCs = 2 #1 or 2 should suffice
flip_threshold = .8
feature_relevance_threshold = 0.01
every_other = 2 #if too much data to store in memory, downsample



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load wavelet transformed data                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------
# Find name tags corresponing to each session
# --------------------------------------------
file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag
session_name_tags = np.load(file_location_concatenated_data + '_session_name_tags.npy')
print(concatenated_data_name_tag)
print(session_name_tags)
session_name_tags_straight = session_name_tags
session_name_tags_upside_down = [x + '_upside_down' for x in session_name_tags]
session_name_tags = np.concatenate((np.array(session_name_tags),np.array(session_name_tags_upside_down)))
session_folders = np.concatenate((np.array(session_name_tags_straight),np.array(session_name_tags_straight)))

if not fetch_extant_LDA:
    # ------------------------------------------
    # Load wavelet-transformed data from file
    # ------------------------------------------
    print('calculating relevant features...')
    feature_used_sum_together = np.zeros(39*39)
    feature_used_std_together = np.zeros(39*39)
    for v in range(len(session_name_tags_straight)):
        file_location_wavelet = file_location + data_folder + analysis_folder + session_folders[v] + '\\' + session_name_tags_straight[v]
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
        
        wavelet_array = np.load(wavelet_file).astype(float)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        coeff_slices = np.load(wavelet_slices_file)
        
        # ---------------------------------------------------
        # Use only features that have non-zero values for LDA
        # ---------------------------------------------------
        feature_used = wavelet_array!=0
        feature_used_sum = np.mean(feature_used,axis=1)
        feature_used_sum_together = feature_used_sum_together + abs(feature_used_sum)
        
        feature_used_std = np.std(wavelet_array,axis=1)
        feature_used_std_together = feature_used_std_together + feature_used_std
    
    relevant_features = (feature_used_sum_together >= feature_relevance_threshold) * (feature_used_std_together >= feature_relevance_threshold) #change to zero to keep more features
    
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
    session_index = []
    for v in range(len(session_name_tags_straight)):
        file_location_wavelet = file_location + data_folder + analysis_folder + session_folders[v] + '\\' + session_name_tags_straight[v]
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
        
        wavelet_array = np.load(wavelet_file).astype(float)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        
        # create resulting output features and mean values to be used in LDA
        wavelet_array_relevant_features_up = np.concatenate((wavelet_array_relevant_features_up, wavelet_array[relevant_features,::every_other].T))
        
        #create session index for displaying particular sessions below
        session_index = np.concatenate((session_index, np.zeros(wavelet_array[relevant_features,::every_other].shape[1])+v),axis = 0)
        
    wavelet_array_relevant_features_up = wavelet_array_relevant_features_up[1:,:]

    print('preparing upside-down features...') 
    wavelet_array_relevant_features_down = np.zeros((1,len(relevant_ind)))
    for v in range(len(session_name_tags_upside_down)):
        file_location_wavelet = file_location + data_folder + analysis_folder + session_folders[v] + '\\' + session_name_tags_upside_down[v]
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
        
        wavelet_array = np.load(wavelet_file).astype(float)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        
        # create resulting output features and mean values to be used in LDA
        wavelet_array_relevant_features_down = np.concatenate((wavelet_array_relevant_features_down, wavelet_array[relevant_features,::every_other].T))
        
    wavelet_array_relevant_features_down = wavelet_array_relevant_features_down[1:,:]
    
    print('appending velocity and rescaling...')
    #append velocity
    
#        position_orientation_velocity_cur[trials_to_analyze_index,0] = frames_cur
#        position_orientation_velocity_cur[:,1] = out_of_bounds_cur
#        position_orientation_velocity_cur[trials_to_analyze_index,2:4] = velocity_cur[:,0:2]
#        position_orientation_velocity_cur[trials_to_analyze_index,4] = orientation_angles_cur
#        position_orientation_velocity_cur[trials_to_analyze_index,5:7] = coordinates_cur
#        position_orientation_velocity_cur[trials_to_analyze_index,7] = disruptions_cur

    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity.npy')
    out_of_bounds = position_orientation_velocity[:,1]
    disruptions = np.ones(len(out_of_bounds)).astype(bool)
    disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])
    
    if every_other > 1:
        position_orientation_velocity = position_orientation_velocity[out_of_bounds==0,:]
        disruptions = disruptions[out_of_bounds==0]
        trials_to_analyze_index = np.array([]).astype(int)
        for v in range(len(session_name_tags_straight)):
            trials_to_analyze_index_cur = find(position_orientation_velocity[:,7]==v)[::every_other]
            trials_to_analyze_index = np.append(trials_to_analyze_index, trials_to_analyze_index_cur)
    else:
        trials_to_analyze_index = out_of_bounds==0
        
    velocity = position_orientation_velocity[trials_to_analyze_index,2:3] #velocity toward head direction    
    disruptions = disruptions[trials_to_analyze_index]

    velocity[disruptions==1,0] = 0 #remove spurious velocities
    mean_vel = np.mean(velocity[:])
    std_vel = np.std(velocity[:])
    greater_than_median_vel_ind = np.squeeze(velocity > np.median(velocity) + std_vel)
    greater_than_sub_median_vel_ind = np.squeeze(velocity >= np.median(velocity) - std_vel)
    less_than_median_vel_ind = np.squeeze(velocity <= np.median(velocity) + std_vel)
    print('only flip frames below ' + str(np.median(velocity) + std_vel) + ' pixels/frame')
    velocity[velocity[:,0]-mean_vel > 8*std_vel,0] = mean_vel + 8*std_vel
    velocity[velocity[:,0]-mean_vel < -8*std_vel,0] = mean_vel - 8*std_vel #saturate velocity below spurious level
    
    #rescale data
    velocity_scaler = sklearn.preprocessing.RobustScaler()
    velocity_scaler.fit(velocity)
    velocity = velocity_scaler.transform(velocity)
    
    up_means = np.mean( wavelet_array_relevant_features_up,axis=0)
    up_stds = np.std( wavelet_array_relevant_features_up,axis=0)
    wavelet_array_relevant_features_up = (wavelet_array_relevant_features_up - up_means) / up_stds
    wavelet_array_relevant_features_down = (wavelet_array_relevant_features_down - up_means) / up_stds
    
    # append data and create labels     
    features_up = np.concatenate((wavelet_array_relevant_features_up,velocity),axis=1)
    features_down = np.concatenate((wavelet_array_relevant_features_down,-1*velocity),axis=1)
    data_for_classifier = np.concatenate((features_up, features_down), axis = 0)
    labels_up = np.ones(features_up.shape[0])
    labels_down = -1 * np.ones(features_down.shape[0])# + 2*greater_than_median_vel_ind
    labels_for_classifier = np.concatenate((labels_up,labels_down), axis = 0 )
    print('')

        
#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Get LDCs                                        -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
    
# ------------------------------------------
# Generate the LDCs for the wavelet features
# ------------------------------------------
for i in range(2):
    if not fetch_extant_LDA:
        print('fitting lda model...')
        lda = LinearDiscriminantAnalysis(solver='lsqr', n_components=num_LDCs, store_covariance=False, tol=0.0001)
        lda.fit(data_for_classifier, labels_for_classifier) # input: (samples, features)
    else: 
        lda = joblib.load(file_location_concatenated_data + '_lda')
        session_index = np.load(file_location_concatenated_data + '_session_index.npy')
        
    accuracy = lda.score(features_up, labels_up)
    
    if i==0: 
        print('round 1/2 performance:')
        flip_up = (lda.predict_proba(features_up)[:,0] > .99)*less_than_median_vel_ind
        flip_down = (lda.predict_proba(features_down)[:,1] > .99)*greater_than_sub_median_vel_ind   
        labels_for_classifier[:len(flip_up)] = labels_up - 2* flip_up
        labels_for_classifier[-len(flip_down):] = labels_down + 2*flip_down
    else:
        print('round 2/2 performance:')
    print('accuracy of ' + str(accuracy))
    print('')
        
    # --------------------------------------------
    # Display the frames classified as upside-down
    # --------------------------------------------
    #calculate probabilities/predictions for a particular session
    print('examining session: ' + session_name_tags[session_to_examine])
    features_lda = features_up[session_index==session_to_examine,:]
    predicted_prob_up = lda.predict_proba(features_up[session_index==session_to_examine,:])
    #predicted_prob_up = lda.predict_proba(features)
    predicted_state_flip = (predicted_prob_up[:,0] > flip_threshold)*less_than_median_vel_ind[session_index==session_to_examine]
    
    flip_ind = find(predicted_state_flip==True)
    keep_ind = find(predicted_state_flip==False)
    print(str(len(flip_ind)) + ' flips out of ' + str(len(predicted_state_flip)) + ' frames')
    
    # Open up the selected data wavelet array
    file_location_session = file_location + data_folder + analysis_folder + session_name_tags_straight[session_to_examine] + '\\' + session_name_tags_straight[session_to_examine]
    wavelet_array_session = np.load(file_location_session + '_wavelet.npy')[:,:,::every_other]
    
    #show in one video the unflipped frames, and in another, those that the model would flip
    for i in range(len(flip_ind)):
        
        #reconstruct image from wavelet transform
        wavelet_up = wavelet_array_session[:,:,keep_ind[i]]
        reconstruction_from_wavelet_up  = reconstruct_from_wavelet(wavelet_up,coeff_slices, level, discard_scale)
        
        wavelet_down = wavelet_array_session[:,:,flip_ind[i]]
        reconstruction_from_wavelet_down  = reconstruct_from_wavelet(wavelet_down,coeff_slices, level, discard_scale)    
        
        cv2.imshow('right-side up',reconstruction_from_wavelet_up.astype(uint8))
        cv2.imshow('up-side down',reconstruction_from_wavelet_down.astype(uint8))
            
        i+=1
            
        if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
            break
            
        if i%500==0:
            print(str(i) + ' out of ' + str(len(flip_ind)) + ' frames complete')
                
        if i >= len(flip_ind) - 1:
            break 
        

# ---------------
# Save the model
# --------------- 
if save_LDA:    
    save_file_model = file_location_concatenated_data + '_lda'
    if os.path.isfile(save_file_model) and do_not_overwrite:
        raise Exception('File already exists') 
    joblib.dump(lda, save_file_model)
    np.save(save_file_model + '_scaling', np.array([[mean_vel],[std_vel],up_means,up_stds]))
    joblib.dump(velocity_scaler, save_file_model + '_velocity_scaling')
    np.save(file_location_concatenated_data + '_session_index',session_index)

    
    
    
    

