'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; import sklearn.decomposition; import os; import warnings; warnings.filterwarnings('once')
from learning_funcs import reconstruct_from_wavelet; from sklearn.externals import joblib


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
num_PCs_to_save = 10
num_PCs_to_examine = 10
feature_relevance_threshold = 0.001

save_PCs = True
examine_PCs = True
cumulative_PCs = False #view reconstruction as the accumulation of PCs vs. using one PC at a time
do_not_overwrite = True
display_frame_rate = 40
end_frame = 10000




#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load wavelet transformed data                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Load wavelet-transformed data from file
# ------------------------------------------

file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag
session_name_tags = np.load(file_location_concatenated_data + '_session_name_tags.npy')
print(session_name_tags)

# these must be parameters taken from original wavelet transform 
level = 5
discard_scale = 4

if os.path.isfile(file_location_wavelet + '_wavelet_relevant_ind_PCA.npy') and True:
    relevant_ind = np.load(file_location_concatenated_data + '_wavelet_relevant_ind_PCA.npy')
else:
    feature_used_sum_together = np.zeros(39*39)
    feature_used_std_together = np.zeros(39*39)
    print('calculating relevant features...')
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
        coeff_slices = np.load(wavelet_slices_file)
        
        # ---------------------------------------------------
        # Use only features that have non-zero values for PCA
        # ---------------------------------------------------
        feature_used = wavelet_array!=0
        feature_used_sum = np.mean(feature_used,axis=1)
        feature_used_sum_together = feature_used_sum_together + abs(feature_used_sum)
        
        feature_used_std = np.std(wavelet_array,axis=1)
        feature_used_std_together = feature_used_std_together + feature_used_std
    
    relevant_features = (feature_used_sum_together >= feature_relevance_threshold) * (feature_used_std_together >= feature_relevance_threshold) #change to zero to keep more features
    
    # also save the index of each of these features
    relevant_ind = find(relevant_features)
    np.save(file_location_concatenated_data + '_wavelet_relevant_ind_PCA.npy', relevant_ind)
   

print(str(sum(relevant_features)) + ' relevant features retained from wavelet transform')
print('')





#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Examine each PC                                  -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

wavelet_array_relevant_features = np.zeros((1,len(relevant_ind)))
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
    wavelet_array_relevant_features = np.concatenate((wavelet_array_relevant_features, wavelet_array[relevant_features,:].T))

wavelet_array_relevant_features = wavelet_array_relevant_features[1:,:]
wavelet_relevant_mean = np.mean(wavelet_array_relevant_features,axis=0)
    
    
if examine_PCs:
    
    # ------------------------------------------
    # Generate the PCs for the wavelet features
    # ------------------------------------------
    print('fitting pca...')
    pca = sklearn.decomposition.PCA(n_components=num_PCs_to_examine, svd_solver = 'arpack') #if too slow, try svd_solver = 'randomized'
    pca.fit(wavelet_array_relevant_features) # input: (samples, features)
    for n_com in range(0,num_PCs_to_examine):

        # -----------------------------------
        # Compute the expansion coefficients
        # -----------------------------------        
        if cumulative_PCs:  # Reconstruct the data based on all the PCs taken so far
            coeffs = pca.transform(wavelet_array_relevant_features).T[0:n_com+1,0:end_frame]
            wavelet_array_relevant_features_recon = pca.inverse_transform(coeffs.T)
        else:               # Reconstruct the data based on only the current PC
            coeffs = pca.transform(wavelet_array_relevant_features).T[n_com:n_com+1,0:end_frame]
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
            empty_wavelet[relevant_ind] = wavelet_array_relevant_features_recon[frame_num,:]
            wavelet = np.reshape(empty_wavelet,(39,39))
             
            #reconstruct image from wavelet transform
            reconstruction_from_wavelet  = reconstruct_from_wavelet(wavelet,coeff_slices, level, discard_scale)
            cv2.imshow('PC / wavelet reconstruction',cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450)))
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
    
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    
#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Save PCs                                        -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


if save_PCs:
    # ------------------------------------------
    # Generate the PCs for the wavelet features
    # ------------------------------------------
    print('saving pca model...')
    pca = sklearn.decomposition.PCA(n_components=num_PCs_to_save, svd_solver = 'arpack') #if too slow, try svd_solver = 'randomized'
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

    



