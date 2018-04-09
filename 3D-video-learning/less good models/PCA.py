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
data_library_name_tag = 'analyze'

examine_PCA_reconstruction = True
examine_PCA_reconstruction_cumulatively = False
do_not_overwrite = True

# ---------------------------
# Select analysis parameters
# ---------------------------
num_PCs_to_create = 10
feature_relevance_threshold = 0.01
modify_relevant_features_from_previous_runs = True; use_relevant_features_from_LDA = False
display_frame_rate = 40






















'''-------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load wavelet transformed data                    -----------------------------------
#------------------------------------------------------------------------------------------------------------------------------------'''



# -----------------------------------------------
# Find data library folder and sessions name tags
# -----------------------------------------------
folder_location_data_library = save_folder_location + data_library_name_tag + '\\'
file_location_data_library = folder_location_data_library + data_library_name_tag
print("saving to " + folder_location_data_library)

session_name_tags = np.load(file_location_data_library + '_session_name_tags.npy')
print(session_name_tags)
session_folders = np.concatenate((np.array(session_name_tags),np.array(session_name_tags)))



# -----------------------------------------------
# Find the indices of important wavelet features
# -----------------------------------------------
coeff_slices = np.load(save_folder_location + 'wavelet_slices.npy')

if not modify_relevant_features_from_previous_runs and os.path.isfile(file_location_data_library + '_relevant_wavelet_features_PCA.npy'):
    relevant_wavelet_features = np.load(file_location_data_library + '_relevant_wavelet_features_PCA.npy')
elif use_relevant_features_from_LDA:
    relevant_wavelet_features = np.load(file_location_data_library + '_relevant_wavelet_features_LDA.npy')
else:
    print('calculating relevant features...')
    wavelet_array_all_sessions = np.zeros((1,39*39))
    wavelet_feature_std_all_sessions = np.zeros(39*39)
    for session in enumerate(session_name_tags):
        print(session[1])
        file_locations_saved_data = glob.glob(save_folder_location + session[1] + '\\' + '*wavelet.npy')
        wavelet_array_session = np.zeros((1,39*39))

        #do so for every video
        for wavelet_video in enumerate(file_locations_saved_data):
            if wavelet_video[1].find('upside') > 0:  # skip upside-down videos
                continue
            wavelet_array = np.load(wavelet_video[1])
            wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
            wavelet_array_session = np.concatenate((wavelet_array_session, wavelet_array.T))
        wavelet_array_all_sessions = np.concatenate((wavelet_array_all_sessions, wavelet_array_session))
    wavelet_array_all_sessions = wavelet_array_all_sessions[1:,:]

    # find the relevant features
    relevant_wavelet_features = (np.std(wavelet_array_all_sessions,axis=0) > feature_relevance_threshold)

    # also save the index of each of these features
    relevant_wavelet_features = np.where(relevant_wavelet_features)[0]
    np.save(file_location_data_library + '_wavelet_relevant_ind_PCA.npy', relevant_wavelet_features)
    print(str(len(relevant_wavelet_features)) + ' relevant features retained from wavelet transform')



''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Examine each PC                                  -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''

wavelet_array_relevant_features = np.zeros((1,len(relevant_wavelet_features)))
for v in range(len(session_name_tags)):
    file_location_wavelet = file_location + data_folder + analysis_folder + session_name_tags[v] + '\\' + session_name_tags[v]
    if os.path.isfile(file_location_wavelet + '_wavelet_corrected.npy'):
        wavelet_file = file_location_wavelet + '_wavelet_corrected.npy'
    else:
        wavelet_file = file_location_wavelet + '_wavelet.npy'
        print('loaded non-flip-corrected data')

    wavelet_slices_file = file_location_wavelet + '_wavelet_slices.npy'
    
    wavelet_array = np.load(wavelet_file)
    wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
    
    # create resulting output features and mean values to be used in PCA
    wavelet_array_relevant_features = np.concatenate((wavelet_array_relevant_features, wavelet_array[relevant_wavelet_features, :].T))

wavelet_array_relevant_features = wavelet_array_relevant_features[1:,:]
wavelet_relevant_mean = np.mean(wavelet_array_relevant_features,axis=0)
    
    
if examine_PCA_reconstruction:
    
    # ------------------------------------------
    # Generate the PCs for the wavelet features
    # ------------------------------------------
    print('fitting pca...')
    pca = sklearn.decomposition.PCA(n_components=num_PCs_to_create, svd_solver ='arpack') #if too slow, try svd_solver = 'randomized'
    pca.fit(wavelet_array_relevant_features) # input: (samples, features)
    for n_com in range(0, num_PCs_to_create):

        # -----------------------------------
        # Compute the expansion coefficients
        # -----------------------------------        
        if examine_PCA_reconstruction_cumulatively:  # Reconstruct the data based on all the PCs taken so far
            coeffs = pca.transform(wavelet_array_relevant_features).T[0:n_com+1,0:1000] # up to only 1000 for speed; if fewer than 1000 frames, decrease this
            wavelet_array_relevant_features_recon = pca.inverse_transform(coeffs.T)
        else:               # Reconstruct the data based on only the current PC
            coeffs = pca.transform(wavelet_array_relevant_features).T[n_com:n_com+1,0:1000]
            wavelet_array_relevant_features_recon = (pca.components_[n_com:n_com+1].T@coeffs).astype(float).T + wavelet_relevant_mean
        
        # -----------------------------------
        # Report PC performance
        # ----------------------------------- 
        print('principal component ' + str(n_com+1)) 
        print(str((100*pca.explained_variance_ratio_[n_com])) + '% var explained by this PC')
        print(str((100*sum(pca.explained_variance_ratio_[0:n_com+1]))) + '% var explained total')
#        recon_error_wavelet = np.linalg.norm(wavelet_array_relevant_features - wavelet_array_relevant_features_recon,axis=1)
#        print('reconstruction error of ' + str(np.mean(recon_error_wavelet)))
        print('')
        
        # ------------------------------------
        # Display PC resonstruction over time
        # ------------------------------------        
        empty_wavelet = np.zeros(39*39)
        
        for frame_num in range(1000):
            empty_wavelet[relevant_wavelet_features] = wavelet_array_relevant_features_recon[frame_num,:]
            wavelet = np.reshape(empty_wavelet,(39,39))
             
            #reconstruct image from wavelet transform
            reconstruction_from_wavelet  = reconstruct_from_wavelet(wavelet,coeff_slices, level, discard_scale)
            cv2.imshow('PC / wavelet reconstruction',cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450)))
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
    
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    
''' -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Save PCs                                        -----------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------
# Generate the PCs for the wavelet features
# ------------------------------------------
print('saving pca model...')
pca = sklearn.decomposition.PCA(n_components=num_PCs_to_create, svd_solver ='arpack') #if too slow, try svd_solver = 'randomized'
pca.fit(wavelet_array_relevant_features) # input: (samples, features)

# -----------------------------------
# Compute the expansion coefficients
# ----------------------------------- 
pca_coeffs = pca.transform(wavelet_array_relevant_features) #input: (samples, features)

# --------------------------------
# Save the model and coefficients
# -------------------------------- 
save_file = file_location_concatenated_data + '_pca_coeffs.npy'
save_file_model = file_location_concatenated_data + '_pca'
if (os.path.isfile(save_file) or os.path.isfile(save_file_model)) and do_not_overwrite:
    raise Exception('File already exists') 
joblib.dump(pca, save_file_model)
np.save(save_file, pca_coeffs)

    



