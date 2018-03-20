'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; import sklearn.decomposition; import os; 
from learning_funcs import reconstruct_from_wavelet; from sklearn.externals import joblib


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
save_vid_name = 'analyze_2D'

file_loc = 'C:\Drive\Video Analysis\data\\'
date = '15.03.2018\\'
mouse_session = 'bj141p2\\'
save_vid_name = 'rectified_norm' # name-tag to be associated with all saved files


# ---------------------------
# Select analysis parameters
# ---------------------------
num_PCs_to_save = 12
num_PCs_to_examine = 12

save_PCs = True
examine_PCs = True
cumulative_PCs = True
do_not_overwrite = False
display_frame_rate = 40

# these must be parameters taken from original wavelet transform 
level = 5
discard_scale = 4



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load wavelet transformed data                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Load wavelet-transformed data from file
# ------------------------------------------
file_loc = file_loc + date + mouse_session + save_vid_name
wavelet_file = file_loc + '_wavelet.npy'
wavelet_slices_file = file_loc + '_wavelet_slices.npy'

wavelet_array = np.load(wavelet_file)
wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
coeff_slices = np.load(wavelet_slices_file)

# ---------------------------------------------------
# Use only features that have non-zero values for PCA
# ---------------------------------------------------
feature_used = wavelet_array!=0
feature_used_sum = np.mean(feature_used,axis=1)
relevant_features = feature_used_sum!=0

# also save the index of each of these features
relevant_ind = find(relevant_features)
np.save(file_loc + '_wavelet_relevant_ind.npy', relevant_ind)

print(str(sum(relevant_features)) + ' relevant features retained from wavelet transform')
print('')

# create resulting output features and mean values to be used in PCA
wavelet_array_relevant_features = wavelet_array[relevant_features,:].T
wavelet_relevant_mean = np.mean(wavelet_array_relevant_features,axis=0)




#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Examine each PC                                  -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


if examine_PCs:
    for n_com in range(0,num_PCs_to_examine):

        # ------------------------------------------
        # Generate the PCs for the wavelet features
        # ------------------------------------------
        pca = sklearn.decomposition.PCA(n_components=n_com+1, svd_solver = 'arpack') #if too slow, try svd_solver = 'randomized'
        pca.fit(wavelet_array_relevant_features) # input: (samples, features)
        
        # -----------------------------------
        # Compute the expansion coefficients
        # -----------------------------------        
        if cumulative_PCs:  # Reconstruct the data based on all the PCs taken so far
            coeffs = pca.transform(wavelet_array_relevant_features).T[0:n_com+1,:]
            wavelet_array_relevant_features_recon = pca.inverse_transform(coeffs.T)
        else:               # Reconstruct the data based on only the current PC
            coeffs = pca.transform(wavelet_array_relevant_features).T[n_com:n_com+1,:]
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
            cv2.imshow('wavelet reconstruction',cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450)))
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
    pca = sklearn.decomposition.PCA(n_components=num_PCs_to_save, svd_solver = 'arpack') #if too slow, try svd_solver = 'randomized'
    pca.fit(wavelet_array_relevant_features) # input: (samples, features)
    
    # -----------------------------------
    # Compute the expansion coefficients
    # ----------------------------------- 
    pca_coeffs = pca.transform(wavelet_array_relevant_features) #input: (samples, features)
    
    # --------------------------------
    # Save the model and coefficients
    # -------------------------------- 
    save_file = file_loc + '_pca_coeffs.npy'
    save_file_model = file_loc + '_pca'
    if (os.path.isfile(save_file) or os.path.isfile(save_file_model)) and do_not_overwrite:
        raise Exception('File already exists') 
    joblib.dump(pca, save_file_model)
    np.save(save_file, pca_coeffs)
    
















