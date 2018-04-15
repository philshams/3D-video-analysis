'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                      Perform PCA on wavelet-transformed mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; import sklearn.decomposition; import os; import warnings; 
from learning_funcs import reconstruct_from_wavelet; from sklearn.externals import joblib; import glob
warnings.filterwarnings('once')

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




examine_PCA_reconstruction = True
examine_PCA_reconstruction_cumulatively = True
do_not_overwrite = True

# ---------------------------
# Select analysis parameters
# ---------------------------
num_PCs_to_create = 10

feature_relevance_threshold = 0.01
modify_relevant_features_from_previous_runs = False; use_relevant_features_from_LDA = False
display_frame_rate = 40






















'''-------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Prepare wavelet transformed data                    -----------------------------------
#------------------------------------------------------------------------------------------------------------------------------------'''



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
coeff_slices = np.load(save_folder_location + 'wavelet_slices.npy')

print('preparing features...')
if not modify_relevant_features_from_previous_runs and os.path.isfile(file_location_data_library + '_relevant_wavelet_features_PCA.npy'):
    relevant_wavelet_features = np.load(file_location_data_library + '_relevant_wavelet_features_PCA.npy')
    new_relevant_wavelet_features = False
elif use_relevant_features_from_LDA:
    relevant_wavelet_features = np.load(file_location_data_library + '_relevant_wavelet_features_LDA.npy')
    new_relevant_wavelet_features = False
else:
    relevant_wavelet_features = np.ones(39*39).astype(bool)
    new_relevant_wavelet_features = True
    print('and calculating relevant features...')
    
    

# ------------------------------------------------------------------------
# for each session, add the wavelet features to the huge array of features
# ------------------------------------------------------------------------
wavelet_array_all_sessions = np.zeros((1, len(relevant_wavelet_features)))
wavelet_feature_std_all_sessions = np.zeros(39 * 39)
session_index = []

for session in enumerate(session_name_tags):
    print(session[1])
    file_locations_saved_data = glob.glob(save_folder_location + session[1] + '\\' + '*wavelet.npy')
    wavelet_array_session = np.zeros((1, len(relevant_wavelet_features)))
    # do so for every video
    for wavelet_video in enumerate(file_locations_saved_data):
        if wavelet_video[1].find('upside') > 0:  # skip upside-down data
            continue
        wavelet_array = np.load(wavelet_video[1])  # .astype(np.float64)
        wavelet_array = np.reshape(wavelet_array, (39 * 39, wavelet_array.shape[2]))
        wavelet_array_session = np.concatenate((wavelet_array_session, wavelet_array[relevant_wavelet_features, :].T))
    wavelet_array_session = wavelet_array_session[1:,:]
    wavelet_array_all_sessions = np.concatenate((wavelet_array_all_sessions, wavelet_array_session))
wavelet_array_all_sessions = wavelet_array_all_sessions[1:, :]



# -------------------------------------------------------------------
# Find the features that vary across time and are therefore relevant
# -------------------------------------------------------------------
if new_relevant_wavelet_features:
    relevant_wavelet_features = (np.std(wavelet_array_all_sessions, axis=0) > feature_relevance_threshold)
    # also save the index of each of these features
    relevant_wavelet_features = np.where(relevant_wavelet_features)[0]
    np.save(file_location_data_library + '_relevant_wavelet_features_PCA.npy', relevant_wavelet_features)    
    wavelet_array_all_sessions = wavelet_array_all_sessions[:, relevant_wavelet_features]
print(str(len(relevant_wavelet_features)) + ' relevant features retained from wavelet transform')

wavelet_relevant_mean = np.mean(wavelet_array_all_sessions, axis=0)
level = 5  # how many different spatial scales are used in wavelet transform
discard_scale = 4


''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Examine each PC                                  -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------''' 

fourcc = cv2.VideoWriter_fourcc(*"XVID")
vid = cv2.VideoWriter('C:\\Drive\\Video Analysis\\data\\calibration_images\\shelter\\pca.avi',fourcc , 40, (450,450), False) 



if examine_PCA_reconstruction:
    
    # ------------------------------------------
    # Generate the PCs for the wavelet features
    # ------------------------------------------
    print('fitting pca...')
    pca = sklearn.decomposition.PCA(n_components=num_PCs_to_create, svd_solver ='arpack') #if too slow, try svd_solver = 'randomized'
    pca.fit(wavelet_array_all_sessions) # input: (samples, features)

    # for each PC:
    for n_com in range(0, num_PCs_to_create):
        # -----------------------------------
        # Compute the expansion coefficients
        # -----------------------------------        
        if examine_PCA_reconstruction_cumulatively:  # Reconstruct the data based on all the PCs taken so far
            coeffs = np.zeros((num_PCs_to_create,1000))
            coeffs[0:n_com + 1,:] = pca.transform(wavelet_array_all_sessions).T[0:n_com + 1, 0:1000]
            wavelet_array_relevant_features_recon = pca.inverse_transform(coeffs.T)
        else:               # Reconstruct the data based on only the current PC
            coeffs = pca.transform(wavelet_array_all_sessions).T[n_com:n_com + 1, 0:1000]
            wavelet_array_relevant_features_recon = (pca.components_[n_com:n_com+1].T@coeffs).astype(float).T + wavelet_relevant_mean
        
        # -----------------------------------
        # Report PC performance
        # ----------------------------------- 
        print('principal component ' + str(n_com+1)) 
        print(str((100*pca.explained_variance_ratio_[n_com])) + '% var explained by this PC')
        print(str((100*sum(pca.explained_variance_ratio_[0:n_com+1]))) + '% var explained total'); print('')
        
        # ------------------------------------
        # Display PC resonstruction over time
        # ------------------------------------        
        empty_wavelet = np.zeros(39*39)
        if n_com == 0 or n_com == 9:
            num_to_see = 400
        else:
            num_to_see = 80
        for frame_num in range(num_to_see):
            empty_wavelet[relevant_wavelet_features] = wavelet_array_relevant_features_recon[frame_num,:]
            wavelet = np.reshape(empty_wavelet,(39,39))
            #reconstruct image from wavelet transform
            reconstruction_from_wavelet  = reconstruct_from_wavelet(wavelet, coeff_slices, level, discard_scale)
            reconstruction_from_wavelet[reconstruction_from_wavelet > 255] = 255
            reconstruction_from_wavelet = cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450))
            cv2.imshow('PC / wavelet reconstruction', reconstruction_from_wavelet)
            vid.write(reconstruction_from_wavelet)
            
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
    
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
vid.release()        
    
''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Save PCs                                        -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''



# Generate the PCs for the wavelet features
print('saving pca model...')
pca = sklearn.decomposition.PCA(n_components=num_PCs_to_create, svd_solver ='arpack') #if too slow, try svd_solver = 'randomized'
pca.fit(wavelet_array_all_sessions) # input: (samples, features)

if os.path.isfile(folder_location_data_library + '_pca') and do_not_overwrite:
        raise Exception('File already exists') 
joblib.dump(pca, file_location_data_library + '_pca')

for session in enumerate(session_name_tags):
    
    wavelet_array = np.load(save_folder_location + session[1] + '\\' + session[1] + '_wavelet.npy')
    wavelet_array = np.reshape(wavelet_array, (39 * 39, wavelet_array.shape[2]))[relevant_wavelet_features,:].T
        
    # Compute the expansion coefficients
    pca_coeffs = pca.transform(wavelet_array) #input: (samples, features)
    
    # Save the coefficients
    pca_file_location = save_folder_location + session[1] + '\\' + session[1] + '_pca_coeffs_' + data_library_name_tag + '.npy'
    np.save(pca_file_location, pca_coeffs)

    



