# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:02:44 2018

@author: SWC
"""
import numpy as np
import cv2
import pywt

#do PCA on library of wavelet-transformed mouse images
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
file_loc = file_loc + date + mouse_session


#first, lets try the simple, analytical solution
def pca(X,n_components = None):
    
    # If no number of component is specified, the function keeps them all
    if n_components is None:
        n_components = X.shape[1]
    
    # Compute mean digit and shift the data
    X_mean = np.mean(X,1)
    X_shifted = X.T - X_mean
    
    # Compute covariance of the data
    cov_X = np.cov(X_shifted.T)
    
    # Compute the eigenvector of the covariance matrix
    w,v = np.linalg.eig(cov_X)
    
    # Retrieve the eigenvectors to return
    components = v[:,0:n_components]
       
    # Returns the transformed data, the components, and the mean digit
    return X_mean, components

def pca_project(Y,X_mean,components):
    # Compute the projection of the input data on the selected components
    Y_shift = Y.T - X_mean
    
    # Compute the expansion coefficients of the data
    components_inv = components.T
    coeffs = components_inv@Y_shift.T
    reconstruction = components@coeffs
    
    return reconstruction.T.astype(np.uint8) + X_mean




#test on mouse images
mouse_array = np.load(file_loc + 'analyzemouse_array.npy').astype(np.uint8)
mouse_array_linear = np.reshape(mouse_array,(20*20,50))
mouse_mean, mouse_components = pca(mouse_array_linear,n_components = 10) #input 400 dim x 50 sample array
mouse_array_recon = pca_project(mouse_array_linear,mouse_mean,mouse_components).astype(np.uint8)
mouse_array_recon = np.reshape(mouse_array_recon.T,(20,20,50))

cv2.imshow('original',cv2.resize(mouse_array[:,:,10],(300,300)))
cv2.imshow('reconstruction',cv2.resize(mouse_array_recon[:,:,10],(300,300)))

#%% 

#do on wavelet transformed arrays
mouse_wavelet = np.load(file_loc + 'wavelet_mouse.npy')
coeff_slices = np.load(file_loc + 'wavelet_slices_mouse.npy')
mouse_wavelet = np.reshape(mouse_wavelet,(39*39,mouse_wavelet.shape[2]))

#only include non-zero features
wavelet_mean = np.mean(mouse_wavelet,axis=1)
relevant_features = wavelet_mean!=0
relevant_ind = find(relevant_features)
print(sum(relevant_features))
mouse_wavelet_relevant = mouse_wavelet[relevant_features,:]


#do PCA on wavelet images
wavelet_mean, wavelet_components = pca(mouse_wavelet,n_components = 1) #input 39*39 dim x 10 sample array
mouse_wavelet_recon = pca_project(mouse_wavelet,wavelet_mean,wavelet_components).T


#%%

#reconstruct wavelet transform
sample_wavelet = np.zeros(39*39)
sample_wavelet[relevant_ind] = mouse_wavelet_recon[relevant_ind,9]
#sample_wavelet = mouse_wavelet[:,9]
wavelet_array = np.reshape(sample_wavelet,(39,39))


#reconstruct image from wavelet transform
level = 5
discard_scale = 4

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

wavelet_recon = pywt.waverec2(coeffs_recon, wavelet='db1').astype(np.uint8)
cv2.imshow('wavelet reconstruction',wavelet_recon)




#%%
#if necessary, do EM algorithm and/or gibbs sampling to get PCs










