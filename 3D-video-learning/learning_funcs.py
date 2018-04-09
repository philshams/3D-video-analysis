'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                Functions for clustering poses and behaviours                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import pywt; import numpy as np; import matplotlib.pyplot as plt; from sklearn import mixture; from hmmlearn import hmm
from sklearn.model_selection import KFold; import pdb; import os; from sklearn.externals import joblib; import cv2; import scipy
''' #run these two lines to reload functions in script without having to start a new kernel
%load_ext autoreload
%autoreload 2
'''

#%% ----------------------------------------------------------------------------------------------------------------------------------
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




#%% ----------------------------------------------------------------------------------------------------------------------------------
def filter_features(data_for_model, filter_length, sigma):
    gauss_filter = np.exp(-np.arange(-filter_length,filter_length+1)**2/sigma**2)  #create filter
    gauss_filter = gauss_filter / sum(gauss_filter) # normalize filter

    data_to_filter_model = np.zeros(data_for_model.shape)
    for pc in range(data_for_model.shape[1]): # apply filter to each ; must be 2D array
        # pad with mirror image
        array_to_filter=np.r_[data_for_model[filter_length:0:-1,pc],data_for_model[:,pc],data_for_model[-1:-filter_length-1:-1,pc]]
        data_to_filter_model[:,pc] = np.convolve(array_to_filter,gauss_filter,mode='valid')        

    data_for_model = data_to_filter_model
    return data_for_model



#%% ----------------------------------------------------------------------------------------------------------------------------------
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


#%% --------------------------------------------------------------------------------------------------------------------
def prepare_data_for_model(pca_coeffs, num_PCs_used, position_orientation_velocity, disruptions, \
                           vel_scaling_factor, turn_scaling_factor, use_speed_as_pseudo_PC, use_angular_speed_as_pseudo_PC, \
                           filter_data_for_model, filter_length, sigma, model_sequence, window_size, windows_to_look_at):
        
    data_for_model = pca_coeffs[:, 0:num_PCs_used]
    # --------------------------------
    # Include velocity as a pseudo-PC
    # --------------------------------
    max_pc = np.max(data_for_model)
    
    if use_speed_as_pseudo_PC:
        speed = position_orientation_velocity[:, 2:3] ** 2 + \
                position_orientation_velocity[:,3:4] ** 2  # speed
        
        # remove spurious velocities
        speed[disruptions == 1,0] = np.mean(speed[disruptions == 0,0])  
    
        # clip outlier speeds
        mean_speed = np.mean(speed[:])
        std_speed = np.std(speed[:])
        speed[speed[:, 0] - mean_speed > 3 * std_speed, 0] = mean_speed + 3 * std_speed  
        
        # rescale speed
        speed_for_model = (speed - np.mean(speed)) / np.max(speed) * max_pc / vel_scaling_factor
    
        # append speed to the rest of the features
        data_for_model = np.append(data_for_model, speed_for_model, axis=1)
    
    # ----------------------------------------
    # Include angular velocity as a pseudo-PC
    # ----------------------------------------
    if use_angular_speed_as_pseudo_PC:
        # get head direction data
        angular_speed_for_model = np.zeros((data_for_model.shape[0], 1))
        head_direction = position_orientation_velocity[:, 4:5]
        last_head_direction = head_direction[:-1, :]; current_head_direction = head_direction[1:, :]
    
        # compute angular speed from head direction data
        angular_speed = np.min(np.concatenate((abs(current_head_direction - last_head_direction),
                                           abs(360 - abs(current_head_direction - last_head_direction))), axis=1), axis=1)
    
        # assume that very large turns in a single frame are spurious, and clip outliers
        angular_speed[angular_speed > 180] = abs(360 - angular_speed[angular_speed > 180])
        angular_speed[angular_speed > 90] = abs(180 - angular_speed[angular_speed > 90])
        angular_speed[angular_speed > 15] = 15
        angular_speed[disruptions[1:]] = 0
    
        # rescale angular speed
        angular_speed = (angular_speed - np.mean(angular_speed)) / np.max(angular_speed) * max_pc / turn_scaling_factor
        angular_speed_for_model[1:, 0] = angular_speed
    
        # double filter this particularly choppy feature
        if filter_data_for_model:
            angular_speed_for_model = filter_features(angular_speed_for_model, filter_length, sigma)
    
        # append angular speed to the rest of the features
        data_for_model = np.append(data_for_model, angular_speed_for_model, axis=1)
    
    # -------------------------------------
    # Smooth features going into the model
    # -------------------------------------
    if filter_data_for_model:
        data_for_model = filter_features(data_for_model, filter_length, sigma)  # apply gaussian smoothing
    data = data_for_model  # use this for the model, unless modeling sequence
    
    # ------------------------------------------------
    # Create array of moving-window-chunked sequences
    # ------------------------------------------------
    if model_sequence:  # add feature chunks preceding and following the frame in question
        data_for_model_sequence = create_sequential_data(data_for_model, window_size, windows_to_look_at)
        data = data_for_model_sequence  # use this for the model, if modeling sequence
        
    return data



#%% ----------------------------------------------------------------------------------------------------------------------------------
def calculate_and_save_model_output(data, model, num_clusters, file_loc, model_type, suffix, do_not_overwrite):
    
    #get probabilities of being in each cluster at each frame
    probabilities = model.predict_proba(data)
    np.save(file_loc+'_probabilities_' + suffix, probabilities)
    
    #get which cluster is chosen at each frame
    chosen_components = model.predict(data)
    np.save(file_loc+'_chosen_components_' + suffix,chosen_components)
    
    #get which cluster is 2nd most likely to be chosen at each frame, and probabilities just for that cluster
    unchosen_probabilities = probabilities # initialize array
    for i in range(num_clusters): #remove chosen cluster's probability
        unchosen_probabilities[chosen_components==i,i]=0 
    unchosen_components = np.argmax(unchosen_probabilities,axis=1) #take the max probability among remaining options
    unchosen_probabilities = np.max(unchosen_probabilities,axis=1) #and report its probability
    np.save(file_loc+'_unchosen_probabilities_' + suffix,unchosen_probabilities)
    
    #get binarized version of chosen_components and unchosen_components, for later analysis
    components_binary = np.zeros((data.shape[0],num_clusters)) #initialize frames x num_clusters array
    unchosen_components_binary = np.zeros((data.shape[0],num_clusters))
    for n in range(num_clusters): #fill with 1 (in that cluster) or 0 (in another cluster)
        components_binary[:,n] = (chosen_components == n)
        unchosen_components_binary[:,n] = (unchosen_components == n)
    np.save(file_loc+'_components_binary_' + suffix,components_binary)
    np.save(file_loc+'_unchosen_components_binary_' + suffix,unchosen_components_binary)
    
    data_for_model_normalized = data / np.max(data)  # set max to 1 for visualization purposes
    np.save(file_loc + '_data_for_' + model_type + '_normalized', data_for_model_normalized)


#%% ----------------------------------------------------------------------------------------------------------------------------------
def create_legend(num_PCs_shown, add_velocity, speed_only, add_change, add_turn):
    legend_entries = []
    for pc in range(num_PCs_shown):
        legend_entries.append('PC' + str(pc+1))
    if add_velocity:
        legend_entries.append('speed')
        if not(speed_only):
            legend_entries.append('turn speed')
    if add_change:
        legend_entries.append('change in pose')
    if add_turn:
        legend_entries.append('turn angle')
    legend = plt.legend((legend_entries))
    return legend_entries


#%% ----------------------------------------------------------------------------------------------------------------------------------
def set_up_PC_cluster_plot(figsize, xlim):
    plt.style.use('classic')
    rolling_fig = plt.figure('PCs and Clusters', figsize=(figsize))
    plt.title('PCs and Clusters over Time')
    plt.xlabel('frame no.')
    plt.ylabel('Feature Amplitude')
    plt.xlim(xlim)
    plt.ylim([-1.05,1.05])
    return rolling_fig


#%% ----------------------------------------------------------------------------------------------------------------------------------
def make_color_array(colors, trajectory_pose_size):
    color_array = np.zeros((trajectory_pose_size, trajectory_pose_size, 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((trajectory_pose_size, trajectory_pose_size)) * colors[c][i] / sum(
                colors[c])
    return color_array

#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_trajectory_indices(windows_to_look_at):
    trajectory_position = [windows_to_look_at]  # get indices to put the various clusters in the right order, below
    offset = 1
    for i in range(2 * windows_to_look_at):
        trajectory_position.append(windows_to_look_at + offset * (i % 2 == 1) - + offset * (i % 2 == 0))
        if (i % 2 == 1):
            offset += 1
    return trajectory_position

#%% ----------------------------------------------------------------------------------------------------------------------------------
def cross_validate_model(data_for_cross_val, model_type, K_range, seed_range, num_components_range, tol):
    
    improvement_thresh = .01
    from_component = num_components_range[0]
    to_component = num_components_range[1]
    test_split = [int(max(K_range)/x) for x in K_range]
    n_components_range = range(from_component,to_component)
    scores_array = np.zeros((len(K_range),len(seed_range),len(n_components_range),2))
    
    k_ind = 0
    for K in K_range:
        for seed in seed_range:
            print(str(K) +'-fold')
            print('seed = ' + str(seed))
                
            #   Choose the number of latent variables by cross-validation
            colors = ['gray', 'turquoise', 'cornflowerblue','navy']
            kf = KFold(n_splits = K,shuffle=True)
            output_matrix = np.zeros((len(n_components_range),4))
            cross_val_matrix = np.zeros((K,4))
            
            
            #Fit a Gaussian mixture with EM, for each no. of components and covariance type
            n = 0
            for n_components in n_components_range:
                k=0
                for train_index, test_index in kf.split(data_for_cross_val):
                    if model_type == 'hmm':
                        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full",algorithm='viterbi',tol=tol,random_state=seed)
                    elif model == 'gmm':
                        model = mixture.GaussianMixture(n_components=n_components,tol=tol,covariance_type='full',random_state=seed)
 
                    model.fit(data_for_cross_val[train_index,:])
                    data_to_test = data_for_cross_val[test_index,:]
                    
                    if test_split[k_ind] != 1:
                        kf_test = KFold(n_splits = test_split[k_ind],shuffle=True)
                        for null_index, test_index_2 in kf_test.split(data_to_test):
                            continue
                        data_to_test = data_to_test[test_index_2,:]
                    
                    probabilities = model.predict_proba(data_to_test)
                    chosen_probabilities = np.max(probabilities,axis=1)
                    cross_val_matrix[k,0] = np.sum(chosen_probabilities>.95) / len(chosen_probabilities) #significant_probs
                    cross_val_matrix[k,1] = np.percentile(chosen_probabilities,10) #median_prob 
                    cross_val_matrix[k,2] = model.score(data_to_test) #score
                    if model_type=='gmm': #calculate BIC by hand, otherwise..?
                        cross_val_matrix[k,3] = model.bic(data_to_test) #bic
                    else:
                        cross_val_matrix[k,3] = len(train_index) * (n_components**2+n) - 2 * np.log(model.score(data_to_test))
                    k+=1
                    
                output_matrix[n,0] = np.mean(cross_val_matrix[:,0]) #significant_probs
                output_matrix[n,1] = np.mean(cross_val_matrix[:,1]) #median_prob 
                output_matrix[n,2] = np.mean(cross_val_matrix[:,2]) #score
                output_matrix[n,3] = np.mean(cross_val_matrix[:,3]) #bic
                
                scores_array[k_ind,seed,n,0] = output_matrix[n,2]
                scores_array[k_ind,seed,n,1] = output_matrix[n,3]
                
                print('for ' + str(n_components) + ' components, there are ' + str(100*output_matrix[n,0]) \
                      + '% signficant groupings and 10 percentile probability ' + str(output_matrix[n,1]))
                print('score of ' + str(output_matrix[n,2]))
                if model_type == 'gmm':
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
            if model_type == 'gmm':
                if bic[0] > 0:
                    plt.bar(X + 0.00, 100* bic / max(bic), color = colors[0], width = 0.6)
                    plt.ylim([np.min(100* bic / max(bic))-.1,100.1])
                else:
                    plt.bar(X + 0.00, 100* bic / abs(min(bic)), color = colors[0], width = 0.6)
                    plt.ylim([-100.1,np.max(100* bic / abs(min(bic)))+.1])
                               
            plt.xticks(n_components_range)
            
            plt.title('Information Content per model (lower is better) - ' + str(K) + '-fold, ' + str(data_for_cross_val.shape[1]) + ' features')
            plt.xlabel('Number of components')
            plt.ylabel('normalized score')
            
            plt.subplot(2,1,2)
            X = np.arange(from_component,to_component)
            plt.bar(X + 0.0, 100*median_probs/ max(median_probs), color = colors[1], width = 0.3)
            plt.bar(X + 0.3, 100*significant_probs / max(significant_probs), color = colors[2], width = 0.3)
            plt.bar(X + 0.6, 100*(scores+150) / (np.max(scores+150)), color = colors[3], width = 0.3)
            plt.ylim([60,103])
                    
            plt.xticks(n_components_range)


            plt.title('scores per model')
            plt.xlabel('Number of components')
            plt.ylabel('normalized score')
            legend = plt.legend(('10th percentile confidence','significant poses','scores'))
            legend.draggable()
          
        k_ind+=1

    return output_matrix
 
#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_biggest_contour(frame):
    _, contours, _ = cv2.findContours(np.array(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cont_count = len(contours)
    
    big_cnt_ind = 0
    if cont_count > 1:
        areas = np.zeros(cont_count)
        for c in range(cont_count):
            areas[c] = cv2.contourArea(contours[c])
        big_cnt_ind = np.argmax(areas)
        
    cnt = contours[big_cnt_ind]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])    
    cy = int(M['m01']/M['m00'])        
    
    return contours, big_cnt_ind, cx, cy, cnt

