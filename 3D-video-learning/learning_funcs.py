# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:34:31 2018

@author: SWC
"""
import pywt
import numpy as np

def reconstruct_from_wavelet(wavelet_array,coeff_slices, level, discard_scale):
    coeffs_recon = [[],[],[],[],[],[]]
    coeffs_extracted = pywt.array_to_coeffs(wavelet_array,coeff_slices)
    
    for i in range(level+1):
        if i==0:
            coeffs_extracted[i] = coeffs_extracted[i]
            coeffs_recon[i] = coeffs_extracted[i]
        elif i < discard_scale:
            coeffs_input = [[],[],[]]
            coeffs_input[0] = coeffs_extracted[i]['da']
            coeffs_input[1] = coeffs_extracted[i]['ad']
            coeffs_input[2] = coeffs_extracted[i]['dd']
            coeffs_recon[i] = coeffs_input
        else:
            coeffs_recon[i] = [None,None,None]

    reconstruction_from_wavelet = pywt.waverec2(coeffs_recon, wavelet='db1')
    return reconstruction_from_wavelet 





#scrap code
#def pca_project(Y,X_mean,components):
#    # Compute the projection of the input data on the selected components
#    Y_shift = Y.T - X_mean
#    
#    # Compute the expansion coefficients of the data
#    coeffs = components@Y_shift.T
#    reconstruction = components.T@coeffs
#    
#    return reconstruction.astype(float).T + X_mean
#
#
#
#
##test on mouse images
#mouse_array = np.load(file_loc + 'analyzemouse_array.npy').astype(np.uint8)
#mouse_array_linear = np.reshape(mouse_array,(20*20,50))
#mouse_mean, mouse_components = pca_manual(mouse_array_linear,n_components = 10) #input 400 dim x 50 sample array
#
#pca = sklearn.decomposition.PCA(n_components=None, svd_solver = 'full')
#pca.fit(mouse_array_linear.T) #input (samples, features)
#
##pca.components_  #components (component#, features)
##pca.explained_variance_ratio_
#
##mouse_array_recon = pca_project(mouse_array_linear,mouse_mean,mouse_components).astype(np.uint8)
#mouse_array_recon = pca_project(mouse_array_linear,mouse_mean,pca.components_).astype(np.uint8)
#mouse_array_recon = np.reshape(mouse_array_recon.T,(20,20,50))
#
#
#
#cv2.imshow('original',cv2.resize(mouse_array[:,:,10],(300,300)))
#cv2.imshow('reconstruction',cv2.resize(mouse_array_recon[:,:,10],(300,300)))
    

##pca code
#    coeffs = pca.components_[0:n_com+1,:]@(wavelet_array_relevant_features - wavelet_relevant_mean).T
#    wavelet_array_relevant_features_recon = (pca.components_[0:n_com+1].T@coeffs).astype(float).T + wavelet_relevant_mean