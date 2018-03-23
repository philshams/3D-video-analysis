'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                                      fit PCA to new data                             -------------------------


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
date = 'baseline_analysis\\'
mouse_session = 'together_for_model\\'
save_vid_names = ['normal_1_0', 'normal_1_1', 'normal_1_2']

pca_name = 'adaboost'
model_file_loc = file_loc + date + mouse_session + pca_name

upside_down = False

# ---------------------------
# Select analysis parameters
# ---------------------------
num_PCs_to_save = 12
num_PCs_to_examine = 12

save_PCs = False
examine_PCs = True
cumulative_PCs = True
do_not_overwrite = True
display_frame_rate = 40

# these must be parameters taken from original wavelet transform 
level = 5
discard_scale = 4



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Examine each PC                                  -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
if upside_down:
    upside_down_suffix = '_upside_down'
else:
    upside_down_suffix = ''
    
relevant_ind = np.load(model_file_loc + '_wavelet_relevant_ind.npy')
coeff_slices = np.load(model_file_loc + '_wavelet_slices.npy')
pca = joblib.load(model_file_loc + '_pca') #load transform info for reconstruction
   
wavelet_array_relevant_features = np.zeros((1,len(relevant_ind)))
for v in range(len(save_vid_names)):
    file_loc_wavelet = file_loc + date + mouse_session + save_vid_names[v]
    wavelet_file = file_loc_wavelet + upside_down_suffix + '_wavelet.npy'
    wavelet_slices_file = file_loc_wavelet + upside_down_suffix + '_wavelet_slices.npy'
    
    wavelet_array = np.load(wavelet_file)
    wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
    
    # create resulting output features and mean values to be used in PCA
    wavelet_array_relevant_features = np.concatenate((wavelet_array_relevant_features, wavelet_array[relevant_ind,:].T))

wavelet_array_relevant_features = wavelet_array_relevant_features[1:,:]
wavelet_relevant_mean = np.mean(wavelet_array_relevant_features,axis=0)
    
    
if examine_PCs:
    print('fitting pca...')
    pca.fit(wavelet_array_relevant_features) # input: (samples, features)
    
    for n_com in range(0,num_PCs_to_examine):
        if cumulative_PCs:
            num_PCs_to_examine = num_PCs_to_save
       
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
            cv2.imshow('PC / wavelet reconstruction',cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450)))
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
            
        if cumulative_PCs:
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
    print('saving pca coeffs...')
    pca.fit(wavelet_array_relevant_features) # input: (samples, features)
    
    # -----------------------------------
    # Compute the expansion coefficients
    # ----------------------------------- 
    pca_coeffs = pca.transform(wavelet_array_relevant_features) #input: (samples, features)
    
    # --------------------------------
    # Save the model and coefficients
    # -------------------------------- 
    save_file = model_file_loc + upside_down_suffix + '_pca_coeffs.npy'
    if (os.path.isfile(save_file)) and do_not_overwrite:
        raise Exception('File already exists') 
    np.save(save_file, pca_coeffs)
