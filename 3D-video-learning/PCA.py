# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:02:44 2018

@author: SWC
"""
#%load_ext autoreload
#%autoreload 2
import numpy as np
import cv2
import sklearn.decomposition
from learning_funcs import reconstruct_from_wavelet
import os

#do PCA on library of wavelet-transformed mouse images
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
file_loc = file_loc + date + mouse_session

save_PCs = True
examine_PCs = False
cumulative_PCs = False

do_not_overwrite = True

#parameters taken from original wavelet transform 
level = 5
discard_scale = 4

#%% load wavelet data

#load wavelet-transformed arrays
wavelet_transformed_array = np.load(file_loc + 'wavelet_mouse.npy')
wavelet_transformed_array = np.reshape(wavelet_transformed_array,(39*39,wavelet_transformed_array.shape[2]))
coeff_slices = np.load(file_loc + 'wavelet_slices_mouse.npy')


#only include non-zero features
feature_used = wavelet_transformed_array!=0
feature_used_sum = np.mean(feature_used,axis=1)
relevant_features = feature_used_sum!=0
relevant_ind = find(relevant_features)
print(str(sum(relevant_features)) + ' relevant features retained from wavelet transform')
print('')
wavelet_array_relevant_features = wavelet_transformed_array[relevant_features,:].T
wavelet_relevant_mean = np.mean(wavelet_array_relevant_features,axis=0)



#%% examine the different principal components

if examine_PCs:
        
    for n_com in range(0,41):
        #do PCA on wavelet images
        #set PCA settings e.g. 'full' vs 'randomized' vs 'arpack'; n_components can also be used to set percent variance explained (0-1)
        pca = sklearn.decomposition.PCA(n_components=n_com+1, svd_solver = 'arpack') 
        pca.fit(wavelet_array_relevant_features) #input is (samples, features)
        
        
        # Compute the expansion coefficients of the data
        if cumulative_PCs:
            coeffs = pca.transform(wavelet_array_relevant_features).T[0:n_com+1,:]
            # Reconstruct the data
            wavelet_array_relevant_features_recon = pca.inverse_transform(coeffs.T)
        else:
            coeffs = pca.transform(wavelet_array_relevant_features).T[n_com:n_com+1,:]
            #coeffs = pca.components_[n_com:n_com+1,:]@(wavelet_array_relevant_features - wavelet_relevant_mean).T
            # Reconstruct the data
            #wavelet_array_relevant_features_recon = pca.inverse_transform(coeffs.T)
            wavelet_array_relevant_features_recon = (pca.components_[n_com:n_com+1].T@coeffs).astype(float).T + wavelet_relevant_mean
        
        #analyze performance 
        print('principal component ' + str(n_com)) 
        print(str((100*pca.explained_variance_ratio_[n_com])) + '% var explained by this PC')
        print(str((100*sum(pca.explained_variance_ratio_[0:n_com+1]))) + '% var explained total')
        recon_error_wavelet = np.linalg.norm(wavelet_array_relevant_features - wavelet_array_relevant_features_recon,axis=1)
        print('reconstruction error of ' + str(mean(recon_error_wavelet)))
        print('')
        
        #build sample wavelet transform array
        sample_wavelet = np.zeros(39*39)
        
        for frame_num in range(1000):
            sample_wavelet[relevant_ind] = wavelet_array_relevant_features_recon[frame_num,:]
            #sample_wavelet[relevant_ind] = wavelet_array_relevant_features[frame_num,:] #ground truth
            wavelet_array = np.reshape(sample_wavelet,(39,39))
             
            #reconstruct image from wavelet transform
            reconstruction_from_wavelet  = reconstruct_from_wavelet(wavelet_array,coeff_slices, level, discard_scale)
            cv2.imshow('wavelet reconstruction',cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450)))
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    
#%% save PCs

if save_PCs:
    #set PCA settings
    #e.g. 'full' vs 'randomized' vs 'arpack'; n_components can also be used to set percent variance explained (0-1)
    pca = sklearn.decomposition.PCA(n_components=9, svd_solver = 'arpack') 
    
    #Apply PCA to data library
    pca.fit(wavelet_array_relevant_features) #input is (samples, features)
    
    # Compute the expansion coefficients of the data to be analyzed, here using the same data as above
    pca_coeffs = pca.transform(wavelet_array_relevant_features) #input is (samples, features)
    
    save_file = file_loc + 'PCA_coeffs_sampledata.npy'
    if os.path.isfile(save_file) and do_not_overwrite:
        raise Exception('File already exists') 
    np.save(save_file, pca_coeffs)
    
    # Reconstruct coeffs into array of wavelet-transformed data
    wavelet_array_relevant_features_recon = pca.inverse_transform(pca_coeffs)

















