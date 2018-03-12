
"""
Created on Tue Feb 27 14:34:31 2018

@author: SWC
#%load_ext autoreload
#%autoreload 2
"""
import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.model_selection import KFold
import pdb
import os
from sklearn.externals import joblib


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


def add_velocity_as_feature(data_for_model, speed_only, velocity, vel_scaling_factor):
    
    max_vel = np.max(abs(velocity[:,0:2]))
    max_pc = np.max(data_for_model)
    
    if not speed_only: #get forward and turn velocity
        velocity_for_model = np.zeros((data_for_model.shape[0],2))
        velocity_for_model[:,:] = velocity[:,0:2] / max_vel * max_pc / vel_scaling_factor
        velocity_for_model[:,1] = np.abs(velocity_for_model[:,1]) - np.mean(np.abs(velocity_for_model[:,1]))
        
    if speed_only: #just get velocity magnitude (speed)
        speed_for_model = np.zeros((data_for_model.shape[0],1))
        speed_for_model[:,0] = np.nan_to_num(np.sqrt(velocity[:,0]**2 + velocity[:,1]**1))
        velocity_for_model = speed_for_model - np.mean(speed_for_model)
        max_speed = np.max(abs(velocity_for_model))
        velocity_for_model = velocity_for_model / max_speed * max_pc / vel_scaling_factor
        
    #append the appropriate velocity array to PCs    
    data_for_model = np.append(data_for_model,velocity_for_model,axis=1)
    
    return data_for_model
    
def add_pose_change_as_feature(data_for_model, vel_scaling_factor, num_PCs_used):
    max_pc = np.max(data_for_model)
    
    ch_ch_ch_ch_changes = np.zeros((data_for_model.shape[0],1))
    ch_ch_ch_ch_changes[:,0] = np.append(0,  #just the norm of the difference in PC space of consecutive frames
        np.linalg.norm(data_for_model[1:,0:num_PCs_used] - data_for_model[:-1,0:num_PCs_used],axis=1))
    ch_ch_ch_ch_changes = ch_ch_ch_ch_changes - np.mean(ch_ch_ch_ch_changes)
    max_change = np.max(ch_ch_ch_ch_changes)
    ch_ch_ch_ch_changes = ch_ch_ch_ch_changes / max_change * max_pc / vel_scaling_factor
    #append the change array to PCs and velocity
    data_for_model = np.append(data_for_model,ch_ch_ch_ch_changes,axis=1)
    return data_for_model

def filter_features(data_for_model, filter_length, sigma):
    gauss_filter = np.exp(-np.arange(-filter_length,filter_length+1)**2/sigma**2)  #create filter
    gauss_filter = gauss_filter / sum(gauss_filter) # normalize filter

    data_to_filter_model = np.zeros(data_for_model.shape)
    for pc in range(data_for_model.shape[1]): # apply filter to each feature
        data_to_filter_model[:,pc] = np.convolve(data_for_model[:,pc],gauss_filter,mode='same')
    data_for_model = data_to_filter_model
    return data_for_model


def create_sequential_data(data_for_model, window_size, windows_to_look_at):
    #smooth data so past and present parts of sequence cover several time points:
    box_filter = np.ones(window_size) / window_size #create boxcar filter
    data_to_filter_model_sequence = np.zeros(data_for_model.shape) #initialize data to be filtered
    for feature in range(data_for_model.shape[1]): # apply boxcar filter to each feature
        data_to_filter_model_sequence[:,feature] = np.convolve(data_for_model[:,feature],box_filter,mode='same')

    #append points from the past and future such that every point is the centre of a sequence
    data_to_concatenate_sequence = data_for_model[window_size*windows_to_look_at:-window_size*windows_to_look_at] #reduce size -- only frames with enough time before and after are considered
    for w in range(windows_to_look_at):
        pre_data_for_model_sequence = data_to_filter_model_sequence[window_size*(windows_to_look_at-(w+1)):-int((windows_to_look_at+w+1)*window_size),:]
        post_data_for_model_sequence = data_to_filter_model_sequence[int((windows_to_look_at+w+1)*window_size):data_for_model.shape[0]
                                                                 -window_size*(windows_to_look_at-(w+1)),:]
        data_to_concatenate_sequence = np.append(data_to_concatenate_sequence, 
                                             np.append(pre_data_for_model_sequence, post_data_for_model_sequence, axis=1), axis=1)   
 
    return data_to_concatenate_sequence

def calculate_and_save_model_output(data, model, num_clusters, file_loc, model_type, suffix, do_not_overwrite):
    save_file_model = file_loc + '_' + model_type + suffix #save model
    if os.path.isfile(save_file_model) and do_not_overwrite:
        raise Exception('File already exists') 
    joblib.dump(model, save_file_model)
    
    #get probabilities of being in each cluster at each frame
    probabilities = model.predict_proba(data)
    np.save(file_loc+'_probabilities' + suffix, probabilities)
    
    #get which cluster is chosen at each frame
    chosen_components = model.predict(data)
    np.save(file_loc+'_chosen_components' + suffix,chosen_components)
    
    #get which cluster is 2nd most likely to be chosen at each frame, and probabilities just for that cluster
    unchosen_probabilities = probabilities # initialize array
    for i in range(num_clusters): #remove chosen cluster's probability
        unchosen_probabilities[chosen_components==i,i]=0 
    unchosen_components = np.argmax(unchosen_probabilities,axis=1) #take the max probability among remaining options
    unchosen_probabilities = np.max(unchosen_probabilities,axis=1) #and report its probability
    np.save(file_loc+'_unchosen_probabilities' + suffix,unchosen_probabilities)
    
    #get binarized version of chosen_components and unchosen_components, for later analysis
    components_binary = np.zeros((data.shape[0],num_clusters)) #initialize frames x num_clusters array
    unchosen_components_binary = np.zeros((data.shape[0],num_clusters))
    for n in range(num_clusters): #fill with 1 (in that cluster) or 0 (in another cluster)
        components_binary[:,n] = (chosen_components == n)
        unchosen_components_binary[:,n] = (unchosen_components == n)
    np.save(file_loc+'_components_binary' + suffix,components_binary)
    np.save(file_loc+'_unchosen_components_binary' + suffix,unchosen_components_binary)


def create_legend(num_PCs_shown, add_velocity, speed_only, add_change):
    legend_entries = []
    for pc in range(num_PCs_shown):
        legend_entries.append('PC' + str(pc+1))
    if add_velocity:
        legend_entries.append('speed')
        if not(speed_only):
            legend_entries.append('turn speed')
    if add_change:
        legend_entries.append('change in pose')
    legend = plt.legend((legend_entries))
    return legend_entries

def set_up_PC_cluster_plot(figsize, xlim):
    plt.style.use('classic')
    plt.figure('PCs and Clusters', figsize=(figsize))
    plt.title('PCs and Clusters over Time')
    plt.xlabel('frame no.')
    plt.ylabel('Feature Amplitude')
    plt.xlim(xlim)
    plt.ylim([-1.05,1.05])

def cross_validate_GMM(data_for_cross_val, dim_range, K_range, seed_range, num_components_range, tol, 
                       add_velocity, velocity_for_model, vel_scaling_factor):
    improvement_thresh = .01
    from_component = num_components_range[0]
    to_component = num_components_range[1]
    test_split = [int(max(K_range)/x) for x in K_range]
    n_components_range = range(from_component,to_component)
    scores_array = np.zeros((len(dim_range),len(K_range),len(seed_range),len(n_components_range),2))
    
    d = 0  #make into a function!
    for dims in dim_range:
        data_to_fit = data_for_cross_val[:,0:dims]
        if add_velocity: # append velocity as pseudo-PC
            data_to_fit = np.append(data_for_cross_val,velocity_for_model,axis=1) 
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
                if bic[0] > 0:
                    plt.bar(X + 0.00, 100* bic / max(bic), color = colors[0], width = 0.6)
                    plt.ylim([np.min(100* bic / max(bic))-.1,100.1])
                else:
                    plt.bar(X + 0.00, 100* bic / abs(min(bic)), color = colors[0], width = 0.6)
                    plt.ylim([-100.1,np.max(100* bic / abs(min(bic)))+.1])
                                   
                plt.xticks(n_components_range)
                
                plt.title('Information Content per model (lower is better) - ' + str(K) + '-fold, ' + str(dims) + ' features')
                plt.xlabel('Number of components')
                plt.ylabel('normalized score')
                
                plt.subplot(2,1,2)
                X = np.arange(from_component,to_component)
                plt.bar(X + 0.0, 100*median_probs/ max(median_probs), color = colors[1], width = 0.3)
                plt.bar(X + 0.3, 100*significant_probs / max(significant_probs), color = colors[2], width = 0.3)
                plt.bar(X + 0.6, 100*(scores+150) / (np.max(scores+150)), color = colors[3], width = 0.3)
                        
                plt.xticks(n_components_range)
                #plt.ylim([np.min(100*(scores+150) / (np.max(scores+150)))-3,103])
                #plt.ylim([np.min(100*significant_probs / max(significant_probs))-3,103])
                #plt.ylim([np.min(100*median_probs / max(median_probs))-3,103])
                #plt.ylim([60,103])
    
                plt.title('scores per model')
                plt.xlabel('Number of components')
                plt.ylabel('normalized score')
                legend = plt.legend(('10th percentile confidence','significant poses','scores'))
                legend.draggable()
              
                
                
#                #choose the correct number of groupings
#                delta_bic = bic[1:] - bic[0:-1]
#                delta_significant_probs = significant_probs[1:] - significant_probs[0:-1]
#                delta_median_probs = median_probs[1:] - median_probs[0:-1]
#                
#                delta_2_bic = delta_bic + np.append(delta_bic[1:],0)
#                delta_2_significant_probs = delta_significant_probs + np.append(delta_significant_probs[1:], 0)
#                delta_2_median_probs = delta_median_probs + np.append(delta_median_probs[1:], 0)
#                
#                delta_3_bic = delta_2_bic + np.append(delta_bic[2:],np.zeros(2))
#                delta_3_significant_probs = delta_2_significant_probs + np.append(delta_significant_probs[2:],np.zeros(2))
#                delta_3_median_probs = delta_2_median_probs + np.append(delta_median_probs[2:],np.zeros(2))
#                
#                satisfactory_n_components = (delta_bic < 0) + (delta_significant_probs > improvement_thresh) + (delta_median_probs > improvement_thresh) + \
#                                            (delta_2_bic < 0) + (delta_2_significant_probs > 2*improvement_thresh) + (delta_2_median_probs > 2*improvement_thresh) + \
#                                            (delta_3_bic < 0) + (delta_3_significant_probs > 3*improvement_thresh) + (delta_2_median_probs > 3*improvement_thresh)
#                
#                num_components_ind = np.nonzero((1-satisfactory_n_components)*n_components_range[:-1])
#                n_components_chosen = n_components_range[num_components_ind[0][0]]
#                
#                print(str(n_components_chosen) + ' components chosen.')
            k_ind+=1
        d+=1
    return output_matrix
    #see_scores_20dim = np.squeeze(scores_array[2,:,:,:,0])
    #see_scores_15dim = np.squeeze(scores_array[1,:,:,:,0])
    #see_scores_10dim = np.squeeze(scores_array[0,:,:,:,0])
    #see_bic_20dim = np.squeeze(scores_array[2,:,:,:,1])
    #see_bic_10dim = np.squeeze(scores_array[0,:,:,:,1])
    #see_bic_15dim = np.squeeze(scores_array[1,:,:,:,1])    
    
    
    
    #super-clustter pie chart
    
#            
## -------------------------------------
## Display Pie Chart for each behaviour
## -------------------------------------
#plt.close('all')
#mean_responsibilities_model = cluster_model_seq.means_
#
#pose_labels = np.array([])
#for p in range(probabilities.shape[1]):
#    pose_labels = np.append(pose_labels, ['pose ' + str(p+1)])
#    
#index_addition1 = probabilities.shape[1]*sequence_based #just look at poses, not past and future onesindex_addition = probabilities.shape[1]*sequence_based*windows_to_look_at
#index_addition2 = probabilities.shape[1]*sequence_based*windows_to_look_at
#for n in range(mean_responsibilities_model.shape[0]):
#    fig = plt.figure('sequence' + str(n+1),figsize=(5,5))
#    plt.pie(mean_responsibilities_model[n,index_addition2:index_addition1+index_addition2] / 
#            np.sum(mean_responsibilities_model[n,index_addition2:index_addition1+index_addition2]), 
#            labels=pose_labels, colors=plot_colors[0:probabilities.shape[1]],shadow=True)
#     
    
    
#    #%% -------------------------------------------------------------------------------------------------------------------------------------
##------------------------              Get distribution over sub-clusters                         --------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------
#
#if display_sub_clusters:
#    # calculate secondary clusters based on threshold set above
#    dual_components = (chosen_components+1)*10+(unchosen_components+1)*(unchosen_probabilities>show_unchosen_cluster)
#    
#    # Create labels for histogram
#    labels = np.array([])
#    for n in range(num_clusters):
#        labels = np.append(labels,(10*(n+1) + arange(num_clusters+1)))
#    
#    # Plot data
#    plt.figure(figsize=(40,7))
#    for n in range(num_clusters):
#        plt.hist(dual_components[(dual_components>=(n+1)*10) * (dual_components<(n+2)*10)],label=labels.astype(str),bins = np.arange(10,61),density = True, color = plot_colors[n])
#    
#    # Format plot
#    plt.xlim([10,(num_clusters+1)*10])
#    plt.ylim([0,.8])
#    plt.xticks(labels)
#    plt.title('Distribution of clusters and sub-clusters')
#    


    # Transition code:
    
    #%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Get Transition Probabilities                         --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
#
## Get counts of (cluster_preceding --> cluster_following) to (i,j) of cluster_counts
#cluster_counts = np.zeros((num_clusters,num_clusters), dtype=int)
#np.add.at(cluster_counts, (chosen_components[:-1], chosen_components[1:]), 1)
#
## Get transition probs (% in column 1 going to row 1,2,3,etc.)
#transition_prob = (cluster_counts.T / np.sum(cluster_counts,axis=1)).T 
#np.nan_to_num(transition_prob,copy=False)
#
## print transition probs
#print((1000*transition_prob).astype(int)/10)
#
#
## Update probabilities based on transitions:
#updated_probabilities = np.zeros(probabilities.shape)
#probabilities_for_update = probabilities
#components_over_time_for_update = components_over_time
#
#for i in range(3):
#    
#    for n in range(probabilities.shape[1]):
#        
#        updated_probabilities[np.append(False,components_over_time_for_update[:-1,n]).astype(bool),:] \
#        = probabilities_for_update[np.append(False,components_over_time_for_update[:-1,n].astype(bool)),:] * transition_prob[:,n]**0
#        
#        updated_probabilities[0,:] = probabilities[0,:]
#        updated_chosen_components = np.argmax(updated_probabilities,axis=1)
#        
#        
#    #get binarized version of chosen_components and unchosen_components, for later analysis
#    updated_components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_clusters))
#    unchosen_components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_clusters))
#    for n in range(num_clusters):
#        updated_components_over_time[:,n] = (updated_chosen_components == n)
#        unchosen_components_over_time[:,n] = (unchosen_components == n)
#        
#    probabilities_for_update = updated_probabilities    
#    components_over_time_for_update = updated_components_over_time
#
#
#
## Get counts of (cluster_preceding --> cluster_following) to (i,j) of cluster_counts
#cluster_counts = np.zeros((num_clusters,num_clusters), dtype=int)
#np.add.at(cluster_counts, (updated_chosen_components[:-1], updated_chosen_components[1:]), 1)
#
## Get transition probs (% in column 1 going to row 1,2,3,etc.)
#transition_prob = (cluster_counts.T / np.sum(cluster_counts,axis=1)).T 
#np.nan_to_num(transition_prob,copy=False)
#
## print transition probs
#print((1000*transition_prob).astype(int)/10)
#print(sum(components_over_time!=updated_components_over_time))
#
#
#
## set plot parameters
#plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
#plt.style.use('classic')
#plt.figure(figsize=(30,10))
#
## plot PCs
#plt.plot(normalized_pca_coeffs[:,0:3])
#if add_velocity:
#    plt.plot(normalized_pca_coeffs[:,-2] * 2, color = 'k',linewidth=2)
#    plt.plot(normalized_pca_coeffs[:,-1] * 2, color = 'gray', linestyle = '--',linewidth=2)
#    
## plot raster of clusters above PCs
#for n in range(num_clusters):
#    component_frames = find(updated_components_over_time[:,n])
#    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
#    
#    unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>show_unchosen_cluster))
#    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=1,marker='|',s=700)
#
## Format plot
#legend = plt.legend(('PC1','PC2','PC3','direct velocity','ortho velocity'))
#legend.draggable()
#plt.title('Principal Components over Time')
#plt.xlabel('frame no.')
#plt.ylabel('PC amplitude')
#plt.xlim([1500,3500])
#plt.ylim([-1,1.05])


#original seq gmm
    # ------------------------------------------------
# Create array of moving-window-chunked sequences
## ------------------------------------------------
#box_filter = np.ones(window_size) / window_size #create boxcar filter
#data_to_fit_gmm_sequence = np.zeros(probabilities.shape)
#probabilities_for_gmm = probabilities
#probabilities_for_gmm[probabilities<0.01] = 0#for efficiency, set small values to 0
#for cluster in range(data_to_fit_gmm_sequence.shape[1]): # apply boxcar filter
#    data_to_fit_gmm_sequence[:,cluster] = np.convolve(probabilities_for_gmm[:,cluster],box_filter,mode='same'
    

#scrap code
    
    
    #empirical pose mean
    
    
#            if src == 0:
#            components_segregated = data_to_fit_gmm[components_over_time[:,n]!=0,:]
#            mean_pca_coeffs = np.mean(components_segregated,axis=0)
#            mean_pca_coeffs_array[:,n,0] = mean_pca_coeffs
#            
#            title = 'empirical wavelet reconstruction cluster ' + str(n)
#            ax1.add_patch(mpl.patches.Rectangle((n+1, -2), 1,4,color=plot_colors[n],alpha=0.3))
#            
#            
            
            
            
    #dual cluster transition probability
    
#    if dual_transitions:
#    ri = dual_components
#    lookup = np.empty((65+1,), dtype=int)
#    lookup[[10,12,13,14,15,20,21,23,24,25,30,31,32,34,35,40,41,42,43,45,50,51,52,53,54]] = np.arange(25)
#    # translate c, r, s to 0, 1, 2
#    rc = lookup[ri]
#    
    #exponential causal filter code:
    
#        exp_filter = np.append(np.exp(np.arange(-filter_length,1)/tau),np.zeros(filter_length))
#    exp_filter = np.flip(exp_filter / sum(exp_filter),axis=0)
#    
#    
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