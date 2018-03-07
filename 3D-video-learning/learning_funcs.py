# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:34:31 2018

@author: SWC
"""
import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.model_selection import KFold


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



def cross_validate_GMM(data_for_cross_val, dim_range, K_range, seed_range, num_components_range, improvement_thresh, tol):
    from_component = num_components_range[0]
    to_component = num_components_range[1]
    test_split = [int(max(K_range)/x) for x in K_range]
    n_components_range = range(from_component,to_component)
    scores_array = np.zeros((len(dim_range),len(K_range),len(seed_range),len(n_components_range),2))
    
    d = 0  #make into a function!
    for dims in dim_range:
        data_to_fit = data_for_cross_val[:,0:dims]
        k_ind = 0
        for K in K_range:
            for seed in seed_range:
                print(K)
                print(seed)
                    
                #   Choose the number of latent variables by cross-validation
                colors = ['gray', 'turquoise', 'cornflowerblue','navy']
                kf = KFold(n_splits = K,shuffle=True)
                output_matrix = np.zeros((len(n_components_range),4))
                cross_val_matrix = np.zeros((K,4))
                
                
                #Fit a Gaussian mixture with EM, for each no. of components and covariance type
                n = 0
                for n_components in n_components_range:
                    k=0
                    for train_index, test_index in kf.split(data_to_fit):
                        gmm = mixture.GaussianMixture(n_components=n_components,tol=.001,covariance_type='full',random_state=seed)
                        gmm.fit(data_to_fit[train_index,:])
                        data_to_test = data_to_fit[test_index,:]
                        
                        if test_split[k_ind] != 1:
                            kf_test = KFold(n_splits = test_split[k_ind],shuffle=True)
                            for null_index, test_index_2 in kf_test.split(data_to_test):
                                continue
                            data_to_test = data_to_test[test_index_2,:]
                        
                        probabilities = gmm.predict_proba(data_to_test)
                        chosen_probabilities = np.max(probabilities,axis=1)
                        cross_val_matrix[k,0] = np.sum(chosen_probabilities>.95) / len(chosen_probabilities) #significant_probs
                        cross_val_matrix[k,1] = np.percentile(chosen_probabilities,10) #median_prob 
                        cross_val_matrix[k,2] = gmm.score(data_to_test) #score
                        cross_val_matrix[k,3] = gmm.bic(data_to_test) #bic
                            
                        k+=1
                        
                    output_matrix[n,0] = np.mean(cross_val_matrix[:,0]) #significant_probs
                    output_matrix[n,1] = np.mean(cross_val_matrix[:,1]) #median_prob 
                    output_matrix[n,2] = np.mean(cross_val_matrix[:,2]) #score
                    output_matrix[n,3] = np.mean(cross_val_matrix[:,3]) #bic
                    
                    scores_array[d,k_ind,seed,n,0] = output_matrix[n,2]
                    scores_array[d,k_ind,seed,n,1] = output_matrix[n,3]
                    
                    print('for ' + str(n_components) + ' components, there are ' + str(100*output_matrix[n,0]) \
                          + '% signficant groupings and 10 percentile probability ' + str(output_matrix[n,1]))
                    print('score of ' + str(output_matrix[n,2]))
                    print('BIC of ' + str(output_matrix[n,3]))
                    print('')
                    n+=1
                     
                significant_probs = output_matrix[:,0]
                median_probs = output_matrix[:,1]
                scores = output_matrix[:,2]
                bic = output_matrix[:,3]
                
                
                #plot the score, significant groupings and lower percentile probability
                plt.figure(figsize=(30,10))
                plt.subplot(2,1,1)
                X = np.arange(from_component,to_component)
                plt.bar(X + 0.00, 100* bic / max(bic), color = colors[0], width = 0.6)
                plt.xticks(n_components_range)
                plt.ylim([np.min(100* bic / max(bic))-.1,100.1])
                plt.title('Information Content per model (lower is better) - ' + str(K) + '-fold, ' + str(dims) + ' features')
                plt.xlabel('Number of components')
                plt.ylabel('normalized score')
                
                plt.subplot(2,1,2)
                X = np.arange(from_component,to_component)
                plt.bar(X + 0.0, 100*median_probs/ max(median_probs), color = colors[1], width = 0.3)
                plt.bar(X + 0.3, 100*significant_probs / max(significant_probs), color = colors[2], width = 0.3)
                plt.bar(X + 0.6, 100*(scores+150) / (np.max(scores+150)), color = colors[3], width = 0.3)
                        
                plt.xticks(n_components_range)
                plt.ylim([np.min(100*(scores+150) / (np.max(scores+150)))-3,103])
                plt.ylim([np.min(100*significant_probs / max(significant_probs))-3,103])
                plt.ylim([np.min(100*median_probs / max(median_probs))-3,103])
                plt.ylim([70,103])
    
                plt.title('scores per model')
                plt.xlabel('Number of components')
                plt.ylabel('normalized score')
                legend = plt.legend(('10th percentile confidence','significant poses','scores'))
                legend.draggable()
                
                
                
                #choose the correct number of groupings
                delta_bic = bic[1:] - bic[0:-1]
                delta_significant_probs = significant_probs[1:] - significant_probs[0:-1]
                delta_median_probs = median_probs[1:] - median_probs[0:-1]
                
                delta_2_bic = delta_bic + np.append(delta_bic[1:],0)
                delta_2_significant_probs = delta_significant_probs + np.append(delta_significant_probs[1:], 0)
                delta_2_median_probs = delta_median_probs + np.append(delta_median_probs[1:], 0)
                
                delta_3_bic = delta_2_bic + np.append(delta_bic[2:],np.zeros(2))
                delta_3_significant_probs = delta_2_significant_probs + np.append(delta_significant_probs[2:],np.zeros(2))
                delta_3_median_probs = delta_2_median_probs + np.append(delta_median_probs[2:],np.zeros(2))
                
                satisfactory_n_components = (delta_bic < 0) + (delta_significant_probs > improvement_thresh) + (delta_median_probs > improvement_thresh) + \
                                            (delta_2_bic < 0) + (delta_2_significant_probs > 2*improvement_thresh) + (delta_2_median_probs > 2*improvement_thresh) + \
                                            (delta_3_bic < 0) + (delta_3_significant_probs > 3*improvement_thresh) + (delta_2_median_probs > 3*improvement_thresh)
                
                num_components_ind = find((1-satisfactory_n_components)*n_components_range[:-1])
                n_components_chosen = n_components_range[num_components_ind[0]]
                
                print(str(n_components_chosen) + ' components chosen.')
            k_ind+=1
        d+=1
    return output_matrix
    #see_scores_20dim = np.squeeze(scores_array[2,:,:,:,0])
    #see_scores_15dim = np.squeeze(scores_array[1,:,:,:,0])
    #see_scores_10dim = np.squeeze(scores_array[0,:,:,:,0])
    #see_bic_20dim = np.squeeze(scores_array[2,:,:,:,1])
    #see_bic_10dim = np.squeeze(scores_array[0,:,:,:,1])
    #see_bic_15dim = np.squeeze(scores_array[1,:,:,:,1])    
    
    
    

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