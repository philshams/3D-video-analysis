'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                    Functions for analyzing behaviour                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import matplotlib.pyplot as plt; import pandas as pd; import os
''' #run these two lines to reload functions in script without having to start a new kernel
%load_ext autoreload
%autoreload 2
'''

#%% ----------------------------------------------------------------------------------------------------------------------------------
def filter_features(data_for_model, filter_length, sigma):
    
    filter_length = int(filter_length / 2)
    
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
def get_speed(speed_filtered_cond, position_orientation_velocity, out_of_bounds, disruptions, filter_length, sigma, session, frames, frames_all):
    speed = np.sqrt(
        position_orientation_velocity[out_of_bounds == 0, 2:3] ** 2 + position_orientation_velocity[out_of_bounds == 0,
                                                                      3:4] ** 2)  # speed
    speed[disruptions == 1, 0] = np.median(speed[disruptions == 0, 0])  # remove spurious speeds
    mean_speed = np.mean(speed[:])
    std_speed = np.std(speed[:])
    speed[speed[:, 0] - mean_speed > 4 * std_speed, 0] = mean_speed + 4 * std_speed  # clip spurious speeds
    

    # enter into full array
    speed_full = np.ones((len(frames_all),1)) * np.nan
    speed_full[frames,0] = speed[:, 0]
    speed_full[0] = 0
    speed_full_panda = pd.Series(speed_full[:,0])
    speed_full[:,0] = speed_full_panda.interpolate()
    
    speed_filtered_full = filter_features(speed_full, filter_length, sigma)
    
    # put it into an array for all sessions of the condition
    speed_filtered_cond[:speed_filtered_full.shape[0], session[0]] = speed_filtered_full[:,0]

    return speed_filtered_full, speed_filtered_cond

#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_acceleration(acceleration_filtered_cond, position_orientation_velocity, out_of_bounds, disruptions, filter_length, sigma, session, frames, frames_all):
    speed = np.sqrt(
        position_orientation_velocity[out_of_bounds == 0, 2:3] ** 2 + position_orientation_velocity[out_of_bounds == 0,
                                                                      3:4] ** 2)  # speed
    speed[disruptions == 1, 0] = np.median(speed[disruptions == 0, 0])  # remove spurious speeds
    mean_speed = np.mean(speed[:])
    std_speed = np.std(speed[:])
    speed[speed[:, 0] - mean_speed > 4 * std_speed, 0] = mean_speed + 4 * std_speed  # clip spurious speeds
    
    acceleration = np.zeros((len(speed),1))
    current_speed = speed[1:]; last_speed = speed[:-1]
    acceleration[1:, :] = abs(current_speed - last_speed)
    acceleration[disruptions == 1, 0] = np.median(acceleration[disruptions == 0, 0])
    acceleration[np.where(disruptions == 1)[0]+1, 0] = np.median(acceleration)
    
    # enter into full array
    acceleration_full = np.ones((len(frames_all),1)) * np.nan
    acceleration_full[frames,0] = acceleration[:, 0]
    acceleration_full[0] = 0
    acceleration_full_panda = pd.Series(acceleration_full[:,0])
    acceleration_full[:,0] = acceleration_full_panda.interpolate()
    
    acceleration_filtered_full = filter_features(acceleration_full, filter_length, sigma)
    
    # put it into an array for all sessions of the condition
    acceleration_filtered_cond[:acceleration_filtered_full.shape[0], session[0]] = acceleration_filtered_full[:,0]

    return acceleration_filtered_full, acceleration_filtered_cond


#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_angular_speed(angular_speed_filtered_cond, position_orientation_velocity, out_of_bounds, disruptions, filter_length, sigma, session, frames, frames_all):
    head_direction = position_orientation_velocity[out_of_bounds == 0, 4:5]

    angular_speed_for_analysis = np.zeros((len(frames), 1))
    last_head_direction = head_direction[:-1, :]
    current_head_direction = head_direction[1:, :]
    angular_speed = np.min(np.concatenate(
        (abs(current_head_direction - last_head_direction), abs(360 - abs(current_head_direction - last_head_direction))),
        axis=1), axis=1)
    angular_speed[angular_speed > 180] = abs(360 - angular_speed[angular_speed > 180])
    angular_speed[angular_speed > 90] = abs(180 - angular_speed[angular_speed > 90])
    angular_speed[angular_speed > 15] = 15  # clip head turn
    angular_speed_for_analysis[1:, 0] = angular_speed
    
    # enter into full array
    angular_speed_full = np.ones((len(frames_all),1)) * np.nan
    angular_speed_full[frames,0] = angular_speed_for_analysis[:, 0]
    angular_speed_full[0] = 0
    angular_speed_full_panda = pd.Series(angular_speed_full[:,0])
    angular_speed_full[:,0] = angular_speed_full_panda.interpolate()
    
    angular_speed_filtered_full = filter_features(angular_speed_full, filter_length, sigma)

    # put it into an array for all sessions of the condition
    angular_speed_filtered_cond[:angular_speed_filtered_full.shape[0], session[0]] = angular_speed_filtered_full[:,0]

    return angular_speed_filtered_full, angular_speed_filtered_cond


#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_proportion_time_in_shelter(in_shelter_filtered_cond, out_of_bounds, filter_length, sigma, session):

    in_shelter = np.zeros((len(out_of_bounds),1))
    in_shelter[:,0] = out_of_bounds==1
    in_shelter_filtered = filter_features(in_shelter, filter_length, sigma)

    #put it into an array for all sessions of the condition
    in_shelter_filtered_cond[:in_shelter_filtered.shape[0],session[0]] = in_shelter_filtered[:,0]
    
    return in_shelter_filtered, in_shelter_filtered_cond


#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_distance_from_centre(distance_from_centre_filtered_cond, position_orientation_velocity, out_of_bounds, centre, shelter, filter_length, sigma, 
                             frames, frames_uncorrupted, frames_all, session, include_shelter = True):

    #calculate distance from centre of arena
    mouse_position = position_orientation_velocity[:,5:7]
    mouse_position_unsheltered = mouse_position[out_of_bounds==0,:]
    if include_shelter:
        mouse_position[out_of_bounds==1,:] = shelter
        mouse_position = mouse_position[np.logical_or(out_of_bounds == 0, out_of_bounds == 1), :]
        frames = frames_uncorrupted
    else:
        mouse_position = mouse_position_unsheltered

    distance_from_centre = np.sqrt((mouse_position[:,0:1] - centre[0])**2 + (mouse_position[:,1:2] - centre[1])**2)

    #enter into full array
    distance_from_centre_full = np.ones((len(frames_all),1))*np.nan
    distance_from_centre_full[frames,0] = distance_from_centre[:,0]
    distance_from_centre_full_panda = pd.Series(distance_from_centre_full[:,0])
    distance_from_centre_full[:,0] = distance_from_centre_full_panda.interpolate()
    
    distance_from_centre_filtered_full = filter_features(distance_from_centre_full, filter_length, sigma)

    #put it into an array for all sessions of the condition
    distance_from_centre_filtered_cond[:distance_from_centre_filtered_full.shape[0],session[0]] = distance_from_centre_filtered_full[:,0]
    
    return distance_from_centre_filtered_full, distance_from_centre_filtered_cond, mouse_position_unsheltered 



#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_position_heat_map(H_cond, mouse_position_unsheltered, out_of_bounds, centre, shelter, heat_map_bins, session, max_minutes_in_heat_map, frame_rate, arena_dilation_factor = 1.3):
    
    try:
        end_frame = sum((out_of_bounds==0)[:max_minutes_in_heat_map * frame_rate * 60])
    except:
        print('max_minutes_in_heat_map greater than current session duration.')
        end_frame = np.inf
        
    distance_from_centre_unsheltered = np.sqrt((mouse_position_unsheltered[:,0] - centre[0])**2 + (mouse_position_unsheltered[:,1] - centre[1])**2)
    mouse_position_unsheltered = mouse_position_unsheltered[distance_from_centre_unsheltered < arena_dilation_factor*(centre[1] - shelter[1]),:]
    mouse_position_unsheltered = mouse_position_unsheltered[:min(end_frame, len(mouse_position_unsheltered))]

    H, x_bins, y_bins = np.histogram2d(mouse_position_unsheltered[:,0], mouse_position_unsheltered[:,1], [heat_map_bins, heat_map_bins], normed=True)
    H_cond[:,:,session[0]] = H
    
    return H, H_cond


#%% ----------------------------------------------------------------------------------------------------------------------------------
def get_behavioural_clusters(in_behavioural_cluster_cumulative_cond, session, model_seq, file_location_data_cur, file_location_data_library,
                              model_name_tag, order_of_clusters, cluster_names, frames, frames_all, num_clusters, filter_length, sigma,threshold_behaviour_duration):
    if model_seq:
        seq = 'seq_'
        add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_location_data_library + '_hmm_settings_' + seq + model_name_tag + '.npy').astype(int)
    else:
        window_size, windows_to_look_at = np.zeros(2).astype(int)
        seq = ''
                    
        
    components_binary = np.load(file_location_data_cur +'_components_binary_' + seq + model_name_tag + '.npy') #[2*window_size*windows_to_look_at:,:]
    frames_model = frames[2*window_size*windows_to_look_at:]
    in_behavioural_cluster = np.zeros((frames_model.shape[0],num_clusters+1))
    in_behavioural_cluster[:,1:num_clusters+1] = components_binary
    
    in_behavioural_cluster_full = np.ones((len(frames_all),components_binary.shape[1]+1))*.0000001
    in_behavioural_cluster_full[frames_model,:] = in_behavioural_cluster
    in_behavioural_cluster_full_filtered = filter_features(in_behavioural_cluster_full, filter_length, sigma)
    in_behavioural_cluster = in_behavioural_cluster_full_filtered[frames_model,:]
    
    #normalize behaviours to sum to 1
    in_behavioural_cluster = np.divide(in_behavioural_cluster.T, np.sum(in_behavioural_cluster,axis=1)).T
    
    #reorder clusters in intuitive order, set above
    in_behavioural_cluster_reorder = np.zeros(in_behavioural_cluster.shape)
#            plot_colors_reorder = np.zeros(len(plot_colors)).astype(str)
    cluster_names_reorder = np.zeros(len(cluster_names)).astype(str)
    for o in enumerate(order_of_clusters): #re-order clusters in intuitive manner
        in_behavioural_cluster_reorder[:,o[0]+1] = in_behavioural_cluster[:,o[1]+1]
#                plot_colors_reorder[o[0]] = plot_colors[o[1]]
        cluster_names_reorder[o[0]] = cluster_names[o[1]]
    in_behavioural_cluster_cumulative = np.zeros(in_behavioural_cluster.shape)
    
    
    for n in range(num_clusters):
        in_behavioural_cluster_cumulative[:,n+1] = in_behavioural_cluster_cumulative[:,n] + in_behavioural_cluster_reorder[:,n+1]      
 
    #enter into full array
    in_behavioural_cluster_cumulative_full = np.ones((len(frames_all),in_behavioural_cluster_cumulative.shape[1]))*np.nan
    for n in range(num_clusters+1):
        in_behavioural_cluster_cumulative_full[frames_model,n] = in_behavioural_cluster_cumulative[:,n]
        in_behavioural_cluster_cumulative_full_panda = pd.Series(in_behavioural_cluster_cumulative_full[:,n])
        in_behavioural_cluster_cumulative_full[:,n] = in_behavioural_cluster_cumulative_full_panda.interpolate() 

    #put it into an array for all sessions of the condition
    in_behavioural_cluster_cumulative_cond[:in_behavioural_cluster_cumulative_full.shape[0],:,session[0]] = in_behavioural_cluster_cumulative_full    
    
    return in_behavioural_cluster_cumulative_full, in_behavioural_cluster_cumulative_cond, components_binary, frames_model, cluster_names_reorder
 
    
#%% ----------------------------------------------------------------------------------------------------------------------------------    
def get_transition(transition_cond, components_binary, out_of_bounds, session, file_location_data_cur, num_clusters, model_seq, model_name_tag, 
                   threshold_behaviour_duration, max_minutes_in_transition_probabilities, frame_rate):
       
    if model_seq:
        seq = 'seq_'
    else:
        seq = ''
    
    try:
        end_frame = sum((out_of_bounds==0)[:max_minutes_in_transition_probabilities * frame_rate * 60])
    except:
        print('max_minutes_in_transition_probabilities greater than current session duration.')
        end_frame = np.inf
        
    chosen_components = np.load(file_location_data_cur +'_chosen_components_' + seq + model_name_tag + '.npy')
    chosen_components = chosen_components[:min(end_frame, len(chosen_components))]

    if threshold_behaviour_duration:
        chosen_components_in_a_row = (chosen_components[:-3] == chosen_components[1:-2]) * \
            (chosen_components[:-3] == chosen_components[2:-1]) * (chosen_components[:-3] == chosen_components[3:])
        four_in_a_row = np.where(chosen_components_in_a_row)[0]
        chosen_components = chosen_components[four_in_a_row]
     
    # Get counts of (cluster_preceding --> cluster_following) to (i,j) of cluster_counts
    cluster_counts = np.zeros((num_clusters,num_clusters), dtype=int)
    np.add.at(cluster_counts, (chosen_components[:-1], chosen_components[1:]), 1)
    
    # Get transition probs (% in column 1 going to row 1,2,3,etc.)
    transition = (cluster_counts.T / np.sum(cluster_counts,axis=1)).T 
    np.nan_to_num(transition,copy=False)
    

    # correct for having removed poses even in long sequences
    if threshold_behaviour_duration:
        for i in range(num_clusters):
            uncorrected_transition = transition[i,i]
            
            avg_duration = 1 / (1 - transition)[i,i]
            transition[i,i] = (avg_duration + 2) / (avg_duration + 3)
            for j in range(num_clusters):
                if i==j:
                    continue
                transition[i,j] = transition[i,j] * (1-transition[i,i]) / (1-uncorrected_transition)
                
    # print transition probs
    print((1000*transition).astype(int)/10)                    
        
    transition_cond[:,:,session[0]] = transition

    return transition, transition_cond
    
#%% ----------------------------------------------------------------------------------------------------------------------------------
def plot_behavioural_clusters(conditions, condition, session, frames_all, minute_multiplier, in_behavioural_cluster_cumulative_full,
                              num_clusters, figure_size, plot_colors, cluster_names_reorder):
    
    #plot behaviour over time, as rectangle with changing area of each colour (cluster) along the session
    behaviour_plot_title = 'behaviour, ' + conditions[condition[0]] + ' session ' + str(session[0]+1)
    fig = plt.figure(behaviour_plot_title, figsize=figure_size)
    ax = plt.subplot(111)
    plt.title(behaviour_plot_title)
    plt.xlabel('time in session (minutes)')
    plt.ylabel('Proportion of time')
    plt.xlim([0,max(frames_all*minute_multiplier)])
    for n in range(num_clusters):
        plt.plot([0],[0],color=plot_colors[num_clusters-(n+1)],linewidth=4)
        plt.fill_between(frames_all*minute_multiplier,in_behavioural_cluster_cumulative_full[:,-(n+1)] , in_behavioural_cluster_cumulative_full[:,-(n+2)],
                                                                                    color=plot_colors[-(n+1)],alpha=0.55)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(np.flip(cluster_names_reorder,0), loc='center left', bbox_to_anchor=(1, 0.8)) # Put a legend to the right of the current axis 
    

#%% ----------------------------------------------------------------------------------------------------------------------------------
def plot_analysis(fig_title, frames_all, minute_multiplier, variable_to_plot, conditions, condition, condition_colors, figure_size,
                  title = 'Avg Speed over Time', x_label = 'time in session (minutes)', y_label = 'Speed (~cm/s)'):


    plt.figure(fig_title, figsize=figure_size)
    for cond in range(len(conditions)):
        plt.plot([0], [0], linewidth=4, color=condition_colors[cond])
        legend = plt.legend(conditions)
        legend.draggable()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(frames_all * minute_multiplier, variable_to_plot, linewidth=4,
             color=condition_colors[condition[0]], alpha=.3)
    plt.pause(.01)
    
    
#%% ----------------------------------------------------------------------------------------------------------------------------------
def plot_analysis_by_condition(fig_title, data_filtered_cond, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_trial):

    speed_filtered_mean = np.nanmean(data_filtered_cond,axis=1)
    speed_filtered_mean = speed_filtered_mean[:len(max_len_frames_all)]
    speed_filtered_sem = np.nanstd(data_filtered_cond,axis=1) / np.sqrt(data_filtered_cond.shape[1])
    speed_filtered_sem = speed_filtered_sem[:len(max_len_frames_all)]

    #plot speed
    plt.figure(fig_title, figsize=figure_size)
    plt.plot(max_len_frames_all*minute_multiplier,speed_filtered_mean,linewidth = 4,color=condition_colors[condition[0]])
    plt.fill_between(max_len_frames_all*minute_multiplier, speed_filtered_mean-speed_filtered_sem, speed_filtered_mean+speed_filtered_sem, color=condition_colors[condition[0]], alpha = .4)
    plt.xlim([0,max_trial])
    plt.pause(.01)    
    
#%% ----------------------------------------------------------------------------------------------------------------------------------
def plot_behaviour_analysis_by_condition(in_behavioural_cluster_cumulative_cond, conditions, condition ,max_len_frames_all, minute_multiplier, plot_colors, num_clusters, cluster_names_reorder, figure_size):    
    
    in_behavioural_cluster_cumulative_mean = np.nanmean(in_behavioural_cluster_cumulative_cond,axis=2)
    in_behavioural_cluster_cumulative_mean = in_behavioural_cluster_cumulative_mean[:len(max_len_frames_all),:]
    in_behavioural_cluster_cumulative_sem = np.nanstd(in_behavioural_cluster_cumulative_cond,axis=2) / np.sqrt(in_behavioural_cluster_cumulative_cond.shape[2])
    in_behavioural_cluster_cumulative_sem = in_behavioural_cluster_cumulative_sem[:len(max_len_frames_all),:]

    #plot behaviour over time, as rectangle with changing area of each colour (cluster) along the session
    fig = plt.figure('behaviour ' + conditions[condition[0]], figsize=figure_size)
    ax = plt.subplot(111)
    plt.title('behaviour, ' + conditions[condition[0]])
    plt.xlabel('time in session (minutes)')
    plt.ylabel('Proportion of time')
    #    plt.xlim([0,max(frames*minute_multiplier)])
    plt.xlim([0,130])
    for n in range(num_clusters):
        plt.plot([0],[0],color=plot_colors[num_clusters-(n+1)],linewidth=4)
        plt.fill_between(max_len_frames_all*minute_multiplier,in_behavioural_cluster_cumulative_mean[:,-(n+1)] , in_behavioural_cluster_cumulative_mean[:,-(n+2)],
                                                                                     color=plot_colors[-(n+1)],alpha=0.5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(np.flip(cluster_names_reorder,0), loc='center left', bbox_to_anchor=(1, 0.8)) # Put a legend to the right of the current axis         
        
    
    
#%% ----------------------------------------------------------------------------------------------------------------------------------        
def plot_transition_probability(transition, avg_over_condition, conditions, condition, session, probability_of_saturation, frame_rate, cluster_names, num_clusters, threshold_behaviour_duration):
            
    plt.style.use('default')
    if avg_over_condition:
        fig = plt.figure('transition probabilities ' + conditions[condition[0]], figsize = (12,8))
        ax = fig.add_subplot(111)
        if threshold_behaviour_duration:
            plt.title('thresholded transition probability & mean duration, ' + conditions[condition[0]])    
        else:
            plt.title('transition probability & mean duration, ' + conditions[condition[0]])    
            
        durations = 1000 / ((1 - transition) * frame_rate)
        transition_sem = np.std(durations,axis=2) / np.sqrt(num_clusters)
        
        for i in range(num_clusters):
            ax.text(i,i+.1,'+/-' + str(int(transition_sem[i,i])), va='center', ha='center',color='gray')
        transition = np.mean(transition,axis=2)
    else:
        fig = plt.figure('transition probabilities ' + conditions[condition[0]] + ' session ' + str(session[0]+1), figsize = (12,8))
        ax = fig.add_subplot(111)
        if threshold_behaviour_duration:
            plt.title('thresholded transition probability & mean duration, ' + conditions[condition[0]] + ' session ' + str(session[0]+1))
        else:
            plt.title('transition probability & mean duration, ' + conditions[condition[0]] + ' session ' + str(session[0]+1))
           
    cax = plt.imshow(transition, vmax=probability_of_saturation,cmap='hot')
    
    cbar = fig.colorbar(cax, ticks=np.arange(0,probability_of_saturation,.01))
    
    
    for i in range(num_clusters):
        ax.text(i,i,str(int(1000 / ((1 - transition[i,i]) * frame_rate))) + 'ms', va='center', ha='center')
    
    ax.set_yticks(np.arange(num_clusters))
    ax.set_yticklabels(cluster_names)
    ax.set_xticks(np.arange(num_clusters))
    ax.set_xticklabels([x[:5] + '...'for x in cluster_names])      
    plt.pause(.01)
    
    
 #%% ----------------------------------------------------------------------------------------------------------------------------------            
def double_edged_sigmoid(x,L,k,x0,double=True):
    
    if double:
        x1 = 180-x0
        y = (L / (1 + np.exp(-k*(x-x0)))) * (L / (1 + np.exp(-k*(x1-x))))
    else:
        y = (L / (1 + np.exp(-k*(x-x0))))
#    plt.figure()
    plt.scatter(x,y)
    return y
             
   
 #%% ----------------------------------------------------------------------------------------------------------------------------------                    
def get_particular_variables(variables_of_interest, save_folder_location, data_library_name_tag, session_name_tag, model_name_tag, model_sequence, clip_suprious_values):

    # -----------
    # find data 
    # -----------
    file_location_data_library = save_folder_location + data_library_name_tag + '\\' + data_library_name_tag
    file_location_data_cur = save_folder_location + session_name_tag + '\\' + session_name_tag
  
    if os.path.isfile(file_location_data_cur  + '_position_orientation_velocity_corrected.npy'):
        position_orientation_velocity_cur = np.load(file_location_data_cur + '_position_orientation_velocity_corrected.npy')
    else:
        position_orientation_velocity_cur = np.load(file_location_data_cur + '_position_orientation_velocity.npy')
    
    # keep track of which indices are in-bounds and valid
    out_of_bounds_cur = position_orientation_velocity_cur[:, 1]
    disruptions_cur = np.ones(len(out_of_bounds_cur)).astype(bool)
    disruptions_cur[1:] = np.not_equal(out_of_bounds_cur[1:], out_of_bounds_cur[:-1])
    disruptions_cur = disruptions_cur[out_of_bounds_cur == 0]
        
    num_frames = position_orientation_velocity_cur.shape[0]
    
    # ---------------------
    # initialize parameters
    # ---------------------
    
    # set size of matrix
    additional_array_dimensions = 0
    if 'position' in variables_of_interest:
        additional_array_dimensions += 1
        
    # initialize behavioural cluster parameters, if applicable    
    if 'behavioural cluster' in variables_of_interest:
        if model_sequence:
            seq = 'seq_'
            add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_location_data_library + '_hmm_settings_' + seq + model_name_tag + '.npy').astype(int)
        else:
            window_size, windows_to_look_at = np.zeros(2).astype(int)
            seq = ''            
        num_clusters = np.load(file_location_data_cur +'_components_binary_' + seq + model_name_tag + '.npy').shape[1]
        additional_array_dimensions += num_clusters-1
        
    # initialize shelter position, if applicable
    if ('velocity relative to shelter' in variables_of_interest) or ('head direction relative to shelter' in variables_of_interest) \
        or ('head turn relative to shelter' in variables_of_interest) or ('distance from shelter' in variables_of_interest):
        if os.path.isfile(file_location_data_cur + '_shelter_roi.npy'):
            shelter_roi = np.load(file_location_data_cur + '_shelter_roi.npy')
            shelter_center = [shelter_roi[0] + shelter_roi[2] / 2, shelter_roi[1] + shelter_roi[3] / 2]
        else:
            raise Exception('please go back create arena and shelter rois from preprocessing script')
    
    
    # ----------------------------------------------------------
    # loop through desired variables, adding them to the matrix
    # ----------------------------------------------------------
    variables_of_interest_matrix = np.ones((num_frames,len(variables_of_interest)+additional_array_dimensions))
    i = 0    
    L = 1
    k = .1
    x0 = 10  
    for variable in enumerate(variables_of_interest):
        if variable[1]=='speed':
            speed = np.sqrt(position_orientation_velocity_cur[out_of_bounds_cur == 0, 2:3] ** 2 + position_orientation_velocity_cur[out_of_bounds_cur == 0,3:4] ** 2)
            speed[disruptions_cur == 1, 0] = np.median(speed[disruptions_cur == 0, 0])  # remove spurious speeds
            
            if clip_suprious_values:
                mean_speed = np.mean(speed[:])
                std_speed = np.std(speed[:])
                speed[speed[:, 0] - mean_speed > 4 * std_speed, 0] = mean_speed + 4 * std_speed  # clip spurious speeds
    
            speed_full = np.ones(num_frames) * np.nan
            speed_full[out_of_bounds_cur==0] = speed[:, 0]
            variables_of_interest_matrix[:,i] = speed_full
            i+=1
                
                
        elif variable[1] == 'velocity relative to shelter':
            
            position = position_orientation_velocity_cur[out_of_bounds_cur==0,5:7]
            velocity_relative_to_shelter = np.zeros(position.shape[0])
            distance_from_shelter = np.sqrt((position[:,0] - shelter_center[0])**2 + (position[:,1] - shelter_center[1])**2)
            last_distance_from_shelter = distance_from_shelter[:-1]; current_distance_from_shelter = distance_from_shelter[1:]; 
            velocity_relative_to_shelter[1:] = last_distance_from_shelter - current_distance_from_shelter
            velocity_relative_to_shelter[disruptions_cur == 1] = np.median(velocity_relative_to_shelter[disruptions_cur == 0])
            
            velocity_relative_to_shelter_full = np.ones(num_frames) * np.nan
            velocity_relative_to_shelter_full[out_of_bounds_cur==0] = velocity_relative_to_shelter
            variables_of_interest_matrix[:,i] = velocity_relative_to_shelter_full
            i+=1            
    
        elif variable[1] == 'head direction relative to shelter' or variable[1] ==  'head turn relative to shelter':
                       
            position = position_orientation_velocity_cur[out_of_bounds_cur==0,5:7]
            shelter_direction = np.angle((shelter_center[0] - position[:,0]) + (position[:,1] - shelter_center[1])*1j,deg=True)
            
            head_direction = position_orientation_velocity_cur[out_of_bounds_cur==0,4]
            #head direction relative to shelter
            head_direction_rel_to_shelter = abs(head_direction - shelter_direction) #left and right counted as equal
            head_direction_rel_to_shelter[head_direction_rel_to_shelter>180] = abs(360 - head_direction_rel_to_shelter[head_direction_rel_to_shelter>180])
            head_direction_rel_to_shelter = head_direction_rel_to_shelter*double_edged_sigmoid(head_direction_rel_to_shelter,L,k,x0,double=False)
    
            if variable[1] == 'head direction relative to shelter':
                head_direction_rel_to_shelter_full = np.ones(num_frames)*np.nan
                head_direction_rel_to_shelter_full[out_of_bounds_cur==0] = head_direction_rel_to_shelter
                variables_of_interest_matrix[:,i] = head_direction_rel_to_shelter_full
                i+=1  
              
                
            elif variable[1] == 'head turn relative to shelter':
                #head direction relative to video frame (up is +90 deg)
                head_direction = position_orientation_velocity_cur[out_of_bounds_cur==0,4:5] 
                # get absolute head turn
                angular_speed_for_clipping = np.zeros(head_direction.shape[0])
                last_head_direction = head_direction[:-1, :]; current_head_direction = head_direction[1:, :]
                angular_speed = np.min(np.concatenate((abs(current_head_direction - last_head_direction),
                                                   abs(360 - abs(current_head_direction - last_head_direction))), axis=1), axis=1)
                # assume that very large turns in a single frame are spurious
                angular_speed[angular_speed > 180] = abs(360 - angular_speed[angular_speed > 180])
                angular_speed[disruptions_cur[1:]] = 0
                angular_speed_for_clipping[1:] = angular_speed
                
                
                #change in head direction relative to shelter
                turn_toward_shelter = np.zeros(len(head_direction_rel_to_shelter))
                last_head_direction = head_direction_rel_to_shelter[:-1]
                current_head_direction = head_direction_rel_to_shelter[1:]
                turn_toward_shelter[1:] = last_head_direction - current_head_direction
                turn_toward_shelter[disruptions_cur] == 0
                turn_toward_shelter = turn_toward_shelter*double_edged_sigmoid(head_direction_rel_to_shelter,L,k,x0,double=True)
    
                #clip turn_toward_shelter to a reasonable value
                turn_toward_shelter[angular_speed_for_clipping > 90] = 0
                turn_toward_shelter[turn_toward_shelter > 90] = 180 - turn_toward_shelter[turn_toward_shelter > 90]
                turn_toward_shelter[turn_toward_shelter < -90] = -180 - turn_toward_shelter[turn_toward_shelter < -90]
                if clip_suprious_values:
                    turn_toward_shelter[turn_toward_shelter > 30] = 30 
                    turn_toward_shelter[turn_toward_shelter < -30] = -30 
                    
                turn_toward_shelter_full = np.ones(num_frames)*np.nan
                turn_toward_shelter_full[out_of_bounds_cur==0] = turn_toward_shelter
                
                variables_of_interest_matrix[:,i] = turn_toward_shelter_full
                i+=1               
                
        elif variable[1] == 'distance from center':
            
            if os.path.isfile(file_location_data_cur + '_arena_roi.npy'):
                arena_roi = np.load(file_location_data_cur + '_arena_roi.npy')
                centre = [arena_roi[0] + arena_roi[2] / 2, arena_roi[1] + arena_roi[3] / 2]
            else:
                raise Exception('please go back create arena and shelter rois from preprocessing script')
            
            #calculate distance from centre of arena, including time spent in shelter
            mouse_position = position_orientation_velocity_cur[:,5:7]
    
            mouse_position[out_of_bounds_cur==1,:] = shelter_center
            frames = np.logical_or(out_of_bounds_cur == 0, out_of_bounds_cur == 1)
            mouse_position = mouse_position[frames, :]
        
            distance_from_centre = np.sqrt((mouse_position[:,0:1] - centre[0])**2 + (mouse_position[:,1:2] - centre[1])**2)
        
            #enter into full array
            distance_from_centre_full = np.ones(num_frames)*np.nan
            distance_from_centre_full[frames] = distance_from_centre[:,0]
            
            variables_of_interest_matrix[:,i] = distance_from_centre_full
            i+=1   

        elif variable[1] == 'distance from shelter':
            #calculate distance from centre of arena, including time spent in shelter
            mouse_position = position_orientation_velocity_cur[:,5:7]
    
            mouse_position[out_of_bounds_cur==1,:] = 0
            frames = np.logical_or(out_of_bounds_cur == 0, out_of_bounds_cur == 1)
            mouse_position = mouse_position[frames, :]
        
            distance_from_shelter = np.sqrt((mouse_position[:,0:1] - shelter_center[0])**2 + (mouse_position[:,1:2] - shelter_center[1])**2)
        
            #enter into full array
            distance_from_shelter_full = np.ones(num_frames)*np.nan
            distance_from_shelter_full[frames] = distance_from_shelter[:,0]
            
            variables_of_interest_matrix[:,i] = distance_from_shelter_full
            i+=1             
       
        elif variable[1] == 'in shelter':
            
            in_shelter = out_of_bounds_cur==1
            variables_of_interest_matrix[:,i] = in_shelter
            i+=1   
            
        elif variable[1] == 'position':
            
            position = position_orientation_velocity_cur[:,5:7]
            variables_of_interest_matrix[:,i:i+2] = position
            i+=2       
            
        elif variable[1] == 'behavioural cluster':
                                        
            components_binary = np.load(file_location_data_cur +'_components_binary_' + seq + model_name_tag + '.npy')
            
            components_binary_full = np.ones((num_frames,num_clusters))*np.nan
            frames = np.where(out_of_bounds_cur==0)[0][2*window_size*windows_to_look_at:]
            components_binary_full[frames,:] = components_binary
            
            variables_of_interest_matrix[:,i:i+num_clusters] = components_binary_full
            i+=num_clusters  
    
    return variables_of_interest_matrix
            
        
        
        
        
