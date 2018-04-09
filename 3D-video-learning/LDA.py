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
session_name_tags = ['loom_1', 'loom_2', 'loom_3', 'loom_4', 'loom_5', 'loom_6',
                     'clicks_1', 'clicks_2', 'clicks_3',
                     'post_clicks_1', 'post_clicks_2', 'post_clicks_3']


data_library_name_tag = 'streamlined'

do_not_overwrite = True


# ---------------------------
# Select analysis parameters
# ---------------------------
session_to_examine = 0
display_frame_rate = 40


feature_relevance_threshold = 0.001; modify_relevant_features_from_previous_runs = True
downsample_every_other = 2 #if too much data to store in memory, downsample
num_LDCs = 2; flip_threshold = .85
























''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Prepare Data for LDA                                  -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''


# -----------------------------------------------
# Find data library folder and sessions name tags
# -----------------------------------------------
folder_location_data_library = save_folder_location + data_library_name_tag + '\\'
if not os.path.isdir(folder_location_data_library):
    os.makedirs(folder_location_data_library)
file_location_data_library = folder_location_data_library + data_library_name_tag
print("saving to " + folder_location_data_library)




# ----------------------------------------------------------------------------------------------------------
# Initialize huge array of wavelet features from all videos, plus the indices of important wavelet features
# ----------------------------------------------------------------------------------------------------------
print('preparing features...')

if modify_relevant_features_from_previous_runs or not os.path.isfile(file_location_data_library + '_relevant_wavelet_features_LDA.npy'):
    print('and calculating relevant features...')
    relevant_wavelet_features  = np.ones(39*39).astype(bool)
    new_relevant_wavelet_features = True
else:
    relevant_wavelet_features = np.load(file_location_data_library + '_relevant_wavelet_features_LDA.npy')
    new_relevant_wavelet_features = False
coeff_slices = np.load(save_folder_location + 'wavelet_slices.npy')

wavelet_array_relevant_features_up = np.zeros((1,len(relevant_wavelet_features)))
wavelet_array_relevant_features_down = np.zeros((1,len(relevant_wavelet_features)))
wavelet_feature_std_all_sessions = np.zeros(39*39)
session_index = []


# -----------------------------------------------------------------------------------------
# for each session, add both the up and down wavelet features to the huge array of features
# -----------------------------------------------------------------------------------------
for session in enumerate(session_name_tags):
    print(session[1])
    file_locations_saved_data = glob.glob(save_folder_location + session[1] + '\\' + session[1] + '_*wavelet.npy')
    print(str(len(file_locations_saved_data)) + ' wavelet files found')
    wavelet_array_session_up = np.zeros((1,len(relevant_wavelet_features)))
    wavelet_array_session_down = np.zeros((1,len(relevant_wavelet_features)))
    #do so for every video
    for wavelet_video in enumerate(file_locations_saved_data): 
        wavelet_array = np.load(wavelet_video[1])  #.astype(np.float64)
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))        
        if wavelet_video[1].find('upside') > 0:  # add this video's array to huge array; downsample if necessary
            wavelet_array_session_down = np.concatenate((wavelet_array_session_down, wavelet_array[relevant_wavelet_features, :].T))
        else:
            wavelet_array_session_up = np.concatenate((wavelet_array_session_up, wavelet_array[relevant_wavelet_features, :].T))
    wavelet_array_relevant_features_down = np.concatenate((wavelet_array_relevant_features_down, wavelet_array_session_down[1::downsample_every_other,:]))
    wavelet_array_relevant_features_up = np.concatenate((wavelet_array_relevant_features_up, wavelet_array_session_up[1::downsample_every_other,:]))
    #create session index for displaying particular sessions below
    session_index = np.concatenate((session_index, np.zeros(wavelet_array_session_up[1::downsample_every_other,:].shape[0]) + session[0]), axis = 0)
wavelet_array_relevant_features_up = wavelet_array_relevant_features_up[1:,:]
wavelet_array_relevant_features_down = wavelet_array_relevant_features_down[1:,:]


# -------------------------------------------------------------------
# Find the features that vary across time and are therefore relevant
# -------------------------------------------------------------------
if new_relevant_wavelet_features:
    relevant_wavelet_features = (np.std(wavelet_array_relevant_features_up,axis=0) > feature_relevance_threshold) + \
                            (np.std(wavelet_array_relevant_features_down,axis=0) > feature_relevance_threshold)
    # also save the index of each of these features
    relevant_wavelet_features = np.where(relevant_wavelet_features)[0]
    np.save(file_location_data_library + '_relevant_wavelet_features_LDA.npy', relevant_wavelet_features)
    print(str(len(relevant_wavelet_features)) + ' relevant features retained from wavelet transform')    
    wavelet_array_relevant_features_up = wavelet_array_relevant_features_up[:,relevant_wavelet_features]
    wavelet_array_relevant_features_down = wavelet_array_relevant_features_down[:,relevant_wavelet_features]


# ---------------------------------------------------------------------------------------------------
# for each session, add position_orientation_velocity, disruptions, and out_of_bounds to huge arrays
# ---------------------------------------------------------------------------------------------------
print('appending velocity and rescaling...')
# create huge position_orientation_velocity array for all sessions
position_orientation_velocity = np.array(([], [], [], [], [], [], [], [])).T
disruptions = []; out_of_bounds = []

for session in enumerate(session_name_tags):
    position_orientation_velocity_cur = np.load(save_folder_location + session[1] + '\\' + session[1] + '_position_orientation_velocity.npy')

    # keep track of which indices are in-bounds and valid
    out_of_bounds_cur = position_orientation_velocity_cur[:,1]
    disruptions_cur = np.ones(len(out_of_bounds_cur)).astype(bool)
    disruptions_cur[1:] = np.not_equal(out_of_bounds_cur[1:],out_of_bounds_cur[:-1])
    disruptions_cur = disruptions_cur[out_of_bounds_cur==0]
    
    position_orientation_velocity_cur = position_orientation_velocity_cur[out_of_bounds_cur==0,:]
    position_orientation_velocity = np.concatenate((position_orientation_velocity, \
                                                    position_orientation_velocity_cur[::downsample_every_other,:]), axis=0)
    
    disruptions.append(disruptions_cur[::downsample_every_other])
    out_of_bounds.append(out_of_bounds_cur[::downsample_every_other])
    
# ----------------------------------
# Add forward velocity as a feature
# ----------------------------------
velocity = position_orientation_velocity[:,2:3] #velocity toward head direction
velocity[disruptions==1,0] = 0 #remove spurious velocities

# if mouse is moving forward, don't flip it -- find indices where mouse is moving forward, etc.
mean_vel = np.mean(velocity[:]); std_vel = np.std(velocity[:])
if isnan(mean_vel):
    raise Exception('invalid velocity array')

greater_than_sub_median_vel_ind = np.squeeze(velocity >= (np.median(velocity) - std_vel))
less_than_super_median_vel_ind = np.squeeze(velocity <= (np.median(velocity) + std_vel))
print('only flip frames below ' + str(np.median(velocity) + std_vel) + ' pixels/frame')

#saturate velocity below spurious level
velocity[(velocity[:,0]-mean_vel) > (6*std_vel),0] = mean_vel + 6*std_vel
velocity[(velocity[:,0]-mean_vel) < (-6*std_vel),0] = mean_vel - 6*std_vel

# -------------
# rescale data
# -------------
velocity_scaler = sklearn.preprocessing.RobustScaler()
velocity_scaler.fit(velocity)
velocity = velocity_scaler.transform(velocity)

wavelet_feature_means = (np.mean(wavelet_array_relevant_features_up, axis=0) + \
                         np.mean(wavelet_array_relevant_features_down, axis=0)) / 2
wavelet_feature_stds = (np.std(wavelet_array_relevant_features_up, axis=0) + \
                        np.std(wavelet_array_relevant_features_down, axis=0)) / 2

wavelet_array_relevant_features_up = (
            (wavelet_array_relevant_features_up - wavelet_feature_means) / wavelet_feature_stds)
wavelet_array_relevant_features_down = (
            (wavelet_array_relevant_features_down - wavelet_feature_means) / wavelet_feature_stds)

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
    accuracy = lda.score(features_up, labels_for_classifier[:len(labels_up)])


    print('round ' + str(i+1) + '/2 performance:')
    print('accuracy of ' + str(accuracy))
    
    # flip the labels of the frames that are likely upside-down
    flip_up = (lda.predict_proba(features_up)[:,0] > .99) * less_than_super_median_vel_ind
    flip_down = (lda.predict_proba(features_down)[:,1] > .99) * greater_than_sub_median_vel_ind
    labels_for_classifier[:len(flip_up)] = labels_up - 2* flip_up
    labels_for_classifier[-len(flip_down):] = labels_down + 2*flip_down

    

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
    wavelet_array_session = np.load(file_location_session + '_wavelet.npy')[:,:,::downsample_every_other]
    
    #show in one video the unflipped frames, and in another, those that the model would flip
    level = 5; discard_scale = 4  # these must be parameters taken from original wavelet transform
    flip_ind_single_video = flip_ind[flip_ind<wavelet_array_session.shape[2]]
    for j in range(len(flip_ind_single_video)):
        #reconstruct images from wavelet transform
        wavelet_up = wavelet_array_session[:,:,keep_ind[j]]; wavelet_down = wavelet_array_session[:,:,flip_ind[j]]
        reconstruction_from_wavelet_up  = reconstruct_from_wavelet(wavelet_up,coeff_slices, level, discard_scale)
        reconstruction_from_wavelet_down  = reconstruct_from_wavelet(wavelet_down,coeff_slices, level, discard_scale)
        cv2.imshow('right-side up',reconstruction_from_wavelet_up.astype(np.uint8))
        cv2.imshow('up-side down',reconstruction_from_wavelet_down.astype(np.uint8))

        if (cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q')):
            break
        if j%500==0:
            print(str(j) + ' out of ' + str(len(flip_ind_single_video)) + ' flip frames complete')


# ---------------
# Save the model
# ---------------
save_file_model = file_location_data_library + '_lda'
if os.path.isfile(save_file_model) and do_not_overwrite:
    raise Exception('File already exists')
joblib.dump(lda, save_file_model)
np.save(save_file_model + '_scaling', np.array([[mean_vel],[std_vel],wavelet_feature_means,wavelet_feature_stds]))
joblib.dump(velocity_scaler, save_file_model + '_velocity_scaling')
np.save(file_location_data_library + '_session_index',session_index)

    
    
    
    

