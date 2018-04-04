'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Plot the Results by Session                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; from matplotlib import pyplot as plt; from scipy import linalg; import os; import sklearn; from sklearn.externals import joblib
from sklearn import mixture; from sklearn.model_selection import KFold; from hmmlearn import hmm; import warnings; warnings.filterwarnings('once')
from learning_funcs import filter_features, create_legend; import cv2; import pandas as pd

#%% -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
# Select model directory
file_location = 'C:\\Drive\\Video Analysis\\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'

concatenated_data_name_tag = 'analyze3' 
session_name_tag = [[['normal_1_0', 'normal_1_1', 'normal_1_2'],
                    ['normal_2_0', 'normal_2_1', 'normal_2_2'],
                    ['normal_3_0', 'normal_3_1', 'normal_3_2'],
                    ['normal_4_0', 'normal_4_1', 'normal_4_2'],
                    ['normal_5_0', 'normal_5_1'],
                    ['normal_6_0', 'normal_6_1', 'normal_6_2']],
                    [['clicks_1_0'],['clicks_2_0'],['clicks_3_0']],
                    [['post_clicks_1_0'],['post_clicks_2_0'],['post_clicks_3_0']]]
conditions = ['normal with loom','clicks with ultrasound', 'post-clicks with ultrasound']
condition_colors = ['black','green','red']

filter_length = 24000
sigma = 12000 #2400 is one minute
num_clusters = 4
#cluster_names = ['sniff around', 'stretch or stand', 'pose turn/shift', 'groom, hunched sniff, rear', 'locomote']

cluster_names = ['look around, turn', 'outstretched sniff', 'locomote',  'groom, hunched sniff, rear']

#order_of_clusters = [4,2,1,0,3]
order_of_clusters = [2,0,1,3]

assert num_clusters == len(cluster_names)
figure_size = (16,8)

model_name_tag = '_seq15_4' #add to save names
model_seq = True
save_data = False
load_data = False
behaviour_only = True

heat_map_bins = 10

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Load and Prepare data                                        --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#find session
plt.close('all')
plt.style.use('classic')
print(str(len(session_name_tag)) + ' conditions found')
file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag 
session_name_tags = np.load(file_location_concatenated_data + '_session_name_tags.npy')

#load data     
if os.path.isfile(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy'):
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy')
else:
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity.npy')
    print('loaded non-flip-corrected data')
    
#take just in-bounds data  
in_bounds_session_index = position_orientation_velocity[position_orientation_velocity[:,1]==0,7]
    
#load analysis
cond_epithet = []
for cond in enumerate(conditions):
    cond_epithet.append(cond[1][0:4])
        
        
#save each array separately in the corresponding folder, and save the cond arrays in the analysis folder    
    
    
for condition in enumerate(session_name_tag):
    print('')
    print('analyzing ' + conditions[condition[0]])
    print(str(len(condition[1])) + ' sessions found')
    session_name_tag_cur = condition[1]
    
    #initialie array for per-condition analysis
    if load_data:
        if not behaviour_only:
            speed_filtered_cond = np.load(file_location_concatenated_data + '_analysis_speed_' + cond_epithet[condition[0]] + '.npy')
            angular_speed_filtered_cond = np.load(file_location_concatenated_data + '_analysis_angular_speed_' + cond_epithet[condition[0]] + '.npy')  
            distance_from_centre_filtered_cond = np.load(file_location_concatenated_data + '_analysis_distance_from_centre_' + cond_epithet[condition[0]] + '.npy')  
            in_shelter_filtered_cond = np.load(file_location_concatenated_data + '_analysis_in_shelter_' + cond_epithet[condition[0]] + '.npy')  
            H_cond = np.load(file_location_concatenated_data + '_analysis_position_' + cond_epithet[condition[0]] + '.npy') 
        in_behavioural_cluster_cumulative_cond = np.load(file_location_concatenated_data + '_analysis_behavioural_cluster_' + cond_epithet[condition[0]] + model_name_tag + '.npy') 
        self_transition_filtered_cond = np.load(file_location_concatenated_data + '_analysis_monotony_' + cond_epithet[condition[0]] + model_name_tag + '.npy')  
        plot_colors_reorder, cluster_names_reorder = np.load(file_location_concatenated_data + '_analysis_settings' + model_name_tag)
        
        max_len_frames_all = np.arange(np.sum(np.isfinite(np.mean(in_shelter_filtered_cond,axis=1))))
    else:
        max_session_length = 600000
        in_behavioural_cluster_cumulative_cond = np.ones((max_session_length,num_clusters+1,len(session_name_tag_cur)))*np.nan
        speed_filtered_cond = np.ones((max_session_length,len(session_name_tag_cur)))*np.nan
        angular_speed_filtered_cond = np.ones((max_session_length,len(session_name_tag_cur)))*np.nan
        distance_from_centre_filtered_cond = np.ones((max_session_length,len(session_name_tag_cur)))*np.nan
        in_shelter_filtered_cond = np.ones((max_session_length,len(session_name_tag_cur)))*np.nan
        self_transition_filtered_cond = np.ones((max_session_length,len(session_name_tag_cur)))*np.nan
        frames_cond = np.ones((max_session_length,3,len(session_name_tag_cur)))*np.nan
        
        H_cond = np.zeros((heat_map_bins,heat_map_bins,len(session_name_tag_cur)))
    
        max_len_frames = []; max_len_frames_uncorrupted = []; max_len_frames_all = []
    
    for session in enumerate(session_name_tag_cur):

        print('')
        file_location_data = file_location + data_folder + analysis_folder + session[1][0] + '\\' + session[1][0]
        trials_to_analyze_index = np.zeros(position_orientation_velocity.shape[0]).astype(bool)
        in_bounds_trials_to_analyze_index = np.zeros(in_bounds_session_index.shape[0]).astype(bool)
        for name_tag in enumerate(session[1]):
            trials_to_analyze_index = trials_to_analyze_index + (position_orientation_velocity[:,7] == find(session_name_tags==name_tag[1]))
            in_bounds_trials_to_analyze_index = in_bounds_trials_to_analyze_index + (in_bounds_session_index==find(session_name_tags==name_tag[1]))
        print('session ' + str(session[0]+1) + ': ' + str(name_tag[0]+1) + ' videos found')
        position_orientation_velocity_session = position_orientation_velocity[trials_to_analyze_index,:]
        
        #find trials in bounds
        out_of_bounds = position_orientation_velocity_session[:,1]
        
        #find trials just after coming back in bounds
        disruptions = np.ones(len(out_of_bounds)).astype(bool)
        disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])
        disruptions = disruptions[out_of_bounds == 0]
        
        #get frame numbers of the trials to analyze
        frames_all = np.arange(position_orientation_velocity_session.shape[0])
        frames_uncorrupted = frames_all[np.logical_or(out_of_bounds==0, out_of_bounds==1)]
        frames = frames_all[out_of_bounds==0]
        minute_multiplier = 1 / (40*60)
        
        
        #%% -------------------------------------------------------------------------------------------------------------------------------------
        #------------------------                    Analyze data                                        --------------------------------------
        #-----------------------------------------------------------------------------------------------------------------------------------------
        
        if not behaviour_only:
            # ------
            # SPEED
            # ------
            #calculate speed
            print('analyzing speed...')
            if load_data:
                speed_filtered_full = speed_filtered_cond[:len(frames_all),session[0]]
            else:
                speed = np.sqrt(position_orientation_velocity_session[out_of_bounds == 0,2:3]**2 + position_orientation_velocity_session[out_of_bounds == 0,3:4]**2) #speed
                speed[disruptions==1,0] = np.mean(speed[disruptions==0,0]) #remove spurious speeds
                mean_speed = np.mean(speed[:])
                std_speed = np.std(speed[:])
                speed[speed[:,0]-mean_speed > 4*std_speed,0] = mean_speed + 4*std_speed #clip spurious speeds
                speed_filtered = filter_features(speed, filter_length, sigma)
                
                #enter into full array
                speed_filtered_full = np.ones(len(frames_all))*np.nan
                speed_filtered_full[frames] = speed_filtered[:,0]
                speed_filtered_full_panda = pd.Series(speed_filtered_full)
                speed_filtered_full[:] = speed_filtered_full_panda.interpolate()
                           
                #put it into an array for all sessions of the condition
                speed_filtered_cond[:speed_filtered_full.shape[0],session[0]] = speed_filtered_full
                
            #plot speed
            plt.figure('speed', figsize=figure_size)
            for cond in range(len(conditions)):
                plt.plot([0],[0],linewidth=4,color = condition_colors[cond])
                legend = plt.legend(conditions)
                legend.draggable()
            plt.title('Avg Speed over Time')
            plt.xlabel('time in session (minutes)')
            plt.ylabel('Speed (~cm/s)')
            plt.plot(frames_all*minute_multiplier,speed_filtered_full/10*40,linewidth = 4,color=condition_colors[condition[0]],alpha=.3)
            plt.pause(.01)
            
            # --------------
            # ANGULAR SPEED
            # --------------
            #calculate angular speed
            print('analyzing angular speed...')
            if load_data:
                angular_speed_filtered_full = angular_speed_filtered_cond[:len(frames_all),session[0]]
            else:        
                head_direction = position_orientation_velocity_session[out_of_bounds==0,4:5]
            
                angular_speed_for_analysis = np.zeros((len(frames),1))
                last_head_direction = head_direction[:-1,:]
                current_head_direction = head_direction[1:,:]
                angular_speed = np.min(np.concatenate( ( abs(current_head_direction - last_head_direction), abs(360-abs(current_head_direction - last_head_direction)) ),axis=1),axis=1)
                angular_speed[angular_speed > 180] = abs(360 - angular_speed[angular_speed > 180])
                angular_speed[angular_speed > 90] = abs(180 - angular_speed[angular_speed > 90])
                angular_speed[angular_speed > 15] = 15 #clip head turn
                angular_speed_for_analysis[1:,0] = angular_speed
                angular_speed_filtered = filter_features(angular_speed_for_analysis, filter_length, sigma)
                
                #enter into full array
                angular_speed_filtered_full = np.ones(len(frames_all))*np.nan
                angular_speed_filtered_full[frames] = angular_speed_filtered[:,0]
                angular_speed_filtered_full_panda = pd.Series(angular_speed_filtered_full)
                angular_speed_filtered_full[:] = angular_speed_filtered_full_panda.interpolate()        
                
                #put it into an array for all sessions of the condition
                angular_speed_filtered_cond[:angular_speed_filtered_full.shape[0],session[0]] = angular_speed_filtered_full
                
            #plot angular speed
            plt.figure('angular speed', figsize=figure_size)
            for cond in range(len(conditions)):
                plt.plot([0],[0],linewidth=4,color = condition_colors[cond])
                legend = plt.legend(conditions)
                legend.draggable()
            plt.title('Avg Angular Speed over Time')
            plt.xlabel('time in session (minutes)')
            plt.ylabel('Angular speed (turns/sec)')
            plt.plot(frames_all*minute_multiplier,angular_speed_filtered_full*40/360,linewidth = 4,color=condition_colors[condition[0]],alpha=.3)
            plt.pause(.01)
            
            # ----------------
            # TIME IN SHELTER
            # ----------------
            #calculate time spent in shelter
            print('analyzing shelter...')
            if load_data:
                in_shelter_filtered = in_shelter_filtered_cond[:len(frames_all),session[0]]
            else:        
                in_shelter = np.zeros((len(out_of_bounds),1))
                in_shelter[:,0] = out_of_bounds==1
                in_shelter_filtered = filter_features(in_shelter, filter_length, sigma)       
                
                #put it into an array for all sessions of the condition
                in_shelter_filtered_cond[:in_shelter_filtered.shape[0],session[0]] = in_shelter_filtered[:,0]
                
            #plot time spent in shelter
            plt.figure('shelter', figsize=figure_size)
            for cond in range(len(conditions)):
                plt.plot([0],[0],linewidth=4,color = condition_colors[cond])
                legend = plt.legend(conditions)
                legend.draggable()
            plt.title('Time Spent in Shelter')
            plt.xlabel('time in session (minutes)')
            plt.ylabel('Proportion of Time')
            plt.plot(frames_all*minute_multiplier,in_shelter_filtered,linewidth = 4,color=condition_colors[condition[0]],alpha=.3)
            plt.pause(.01)
            
        # ---------
        # POSITION
        # ---------
        print('analyzing position...')
        #report centre of arena
        if os.path.isfile(file_location_data + '_centre_shelter.npy'):
            centre_shelter = np.load(file_location_data + '_centre_shelter.npy')
            centre = centre_shelter[0]
            shelter = centre_shelter[1]
        else:
            arena_vid = cv2.VideoCapture(file_location_data + '_normalized_video.avi')
            ret, frame = arena_vid.read()
            print('select centre of arena')
            centre_roi = cv2.selectROI(frame[:,:,1])
            centre = [centre_roi[0]+centre_roi[2]/2, centre_roi[1]+centre_roi[3]/2]
            
            #report centre of shelter
            print('select centre of shelter')
            shelter_roi = cv2.selectROI(frame[:,:,1])
            print('thank you')
            shelter = [shelter_roi[0]+shelter_roi[2]/2, shelter_roi[1]+shelter_roi[3]/2]
            np.save(file_location_data + '_centre_shelter.npy',[centre, shelter])
                
            
                    
        if load_data:
            distance_from_centre_filtered_full = distance_from_centre_filtered_cond[:len(frames_all),session[0]]
            
        else:            
            #calculate distance from centre of arena
            mouse_position = position_orientation_velocity_session[:,5:7]
            mouse_position_unsheltered = mouse_position[out_of_bounds==0,:]
#                mouse_position[out_of_bounds==1,:] = shelter
            mouse_position = mouse_position_unsheltered
#                mouse_position = mouse_position[np.logical_or(out_of_bounds==0, out_of_bounds==1),:]
            distance_from_centre = np.sqrt((mouse_position[:,0:1] - centre[0])**2 + (mouse_position[:,1:2] - centre[1])**2)
            distance_from_centre_filtered = filter_features(distance_from_centre, filter_length, sigma)
            
            #enter into full array
            distance_from_centre_filtered_full = np.ones(len(frames_all))*np.nan
#                distance_from_centre_filtered_full[frames_uncorrupted] = distance_from_centre_filtered[:,0]
            distance_from_centre_filtered_full[frames] = distance_from_centre_filtered[:,0]
            distance_from_centre_filtered_full_panda = pd.Series(distance_from_centre_filtered_full)
            distance_from_centre_filtered_full[:] = distance_from_centre_filtered_full_panda.interpolate()        
                   
            #put it into an array for all sessions of the condition
            distance_from_centre_filtered_cond[:distance_from_centre_filtered_full.shape[0],session[0]] = distance_from_centre_filtered_full

        #plot distance from centre      
        plt.figure('distance from centre', figsize=figure_size)
        for cond in range(len(conditions)):
            plt.plot([0],[0],linewidth=4,color = condition_colors[cond])
            legend = plt.legend(conditions)
            legend.draggable()
        plt.title('Distance from Centre of Arena')
        plt.xlabel('time in session (minutes)')
        plt.ylabel('Distance (~cm)')
        plt.plot(frames_all*minute_multiplier,distance_from_centre_filtered_full/10,linewidth = 4,color=condition_colors[condition[0]],alpha=.3)
        plt.pause(.01)
        
        #generate heat map
        if load_data:
            H = np.squeeze(H_cond[:,:,session[0]])
        else:
            distance_from_centre_unsheltered = np.sqrt((mouse_position_unsheltered[:,0] - centre[0])**2 + (mouse_position_unsheltered[:,1] - centre[1])**2)
            mouse_position_unsheltered = mouse_position_unsheltered[distance_from_centre_unsheltered < 1.3*(centre[1] - shelter[1]),:]
            
            H, x_bins, y_bins = np.histogram2d(mouse_position_unsheltered[:,0], mouse_position_unsheltered[:,1], [heat_map_bins, heat_map_bins], normed=True)
            H_cond[:,:,session[0]] = H
        
        plt.figure('position heat map ' + conditions[condition[0]] + ' session ' + str(session[0]+1))
        plt.imshow(H.T, vmax=np.percentile(H,97))
        plt.title('position heat map ' + conditions[condition[0]] + ' session ' + str(session[0]+1))
            
        
        
        # ----------
        # BEHAVIOUR
        # ----------
        print('analyzing behaviour...')
        plot_colors = ['mediumseagreen','blue','purple','red','darkorange']
        plot_colors = plot_colors[:num_clusters]
        
        if load_data:
            in_behavioural_cluster_cumulative_full = in_behavioural_cluster_cumulative_cond[:len(frames_all),:,session[0]]
        else:
            #calculate relative amound of time spent in each behavioural cluster (including shelter)
            
            if model_seq:
                add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_location_concatenated_data + '_hmm_settings' + model_name_tag + '.npy').astype(int)
            else:
                window_size, windows_to_look_at = np.zeros(2).astype(int)
            in_bounds_trials_to_analyze_index_model = in_bounds_trials_to_analyze_index[2*window_size*windows_to_look_at:].copy()
            if session[0]>0 or condition[0]>0:
                in_bounds_trials_to_analyze_index_model[in_bounds_trials_to_analyze_index_model] = np.concatenate((np.zeros(2*window_size*windows_to_look_at),np.ones(len(in_bounds_trials_to_analyze_index_model[in_bounds_trials_to_analyze_index_model])-2*window_size*windows_to_look_at)))
            components_binary = np.load(file_location_concatenated_data+'_components_binary' + model_name_tag + '.npy')[in_bounds_trials_to_analyze_index_model,:]
            frames_model = frames[2*window_size*windows_to_look_at:]
            in_behavioural_cluster = np.zeros((frames_model.shape[0],num_clusters+1))
            in_behavioural_cluster[:,1:num_clusters+1] = components_binary
            in_behavioural_cluster = filter_features(in_behavioural_cluster, filter_length, sigma)
            
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
                
        
        #plot behaviour over time, as rectangle with changing area of each colour (cluster) along the session
        behaviour_plot_title = 'behaviour, ' + conditions[condition[0]] + ' session ' + str(session[0]+1)
        fig = plt.figure(behaviour_plot_title, figsize=figure_size)
        ax = plt.subplot(111)
        plt.title(behaviour_plot_title)
        plt.xlabel('time in session (minutes)')
        plt.ylabel('Proportion of time')
        plt.xlim([0,max(frames*minute_multiplier)])
        for n in range(num_clusters):
            plt.plot([0],[0],color=plot_colors[num_clusters-(n+1)],linewidth=4)
            plt.fill_between(frames_all*minute_multiplier,in_behavioural_cluster_cumulative_full[:,-(n+1)] , in_behavioural_cluster_cumulative_full[:,-(n+2)], 
                                                                                        color=plot_colors[-(n+1)],alpha=0.55)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(np.flip(cluster_names_reorder,0), loc='center left', bbox_to_anchor=(1, 0.8)) # Put a legend to the right of the current axis
        
        
        #percent self-transition
        if load_data:
            self_transition_filtered_full = self_transition_filtered_cond[:len(frames_all),session[0]]
        else:
            self_transition = np.zeros((components_binary.shape[0],1))
            last_components_binary = components_binary[:-1]
            self_transition[1:,0] = np.sum((last_components_binary + components_binary[1:])== 2,axis = 1) 
            self_transition_filtered = filter_features(self_transition, filter_length, sigma)
        
            #enter into full array
            self_transition_filtered_full = np.ones(len(frames_all))*np.nan
            self_transition_filtered_full[frames_model] = self_transition_filtered[:,0]
            self_transition_filtered_full_panda = pd.Series(self_transition_filtered_full)
            self_transition_filtered_full[:] = self_transition_filtered_full_panda.interpolate()         
            
            #put it into an array for all sessions of the condition
            self_transition_filtered_cond[:self_transition_filtered_full.shape[0],session[0]] = self_transition_filtered_full

        
        #plot self-transitions      
        plt.figure('self-transitions', figsize=figure_size)
        for cond in range(len(conditions)):
            plt.plot([0],[0],linewidth=4,color = condition_colors[cond])
            legend = plt.legend(conditions)
            legend.draggable()
        plt.title('Behavioural Monotony')
        plt.xlabel('time in session (minutes)')
        plt.ylabel('Proportion self-transitions')
        plt.plot(frames_all*minute_multiplier,self_transition_filtered_full,linewidth = 4,color=condition_colors[condition[0]],alpha=.3)        
        
        
        
#        if len(frames) > len(max_len_frames):
#            max_len_frames = frames
#        if len(frames_uncorrupted) > len(max_len_frames_uncorrupted):
#            max_len_frames_uncorrupted = frames_uncorrupted
        if len(frames_all) > len(max_len_frames_all) and not load_data:
            max_len_frames_all = frames_all
            
            
    if save_data:
        if not behaviour_only:
            np.save(file_location_concatenated_data + '_analysis_speed_' + cond_epithet[condition[0]], speed_filtered_cond)
            np.save(file_location_concatenated_data + '_analysis_angular_speed_' + cond_epithet[condition[0]], angular_speed_filtered_cond)  
            np.save(file_location_concatenated_data + '_analysis_distance_from_centre_' + cond_epithet[condition[0]], distance_from_centre_filtered_cond)  
            np.save(file_location_concatenated_data + '_analysis_in_shelter_' + cond_epithet[condition[0]], in_shelter_filtered_cond)  
            np.save(file_location_concatenated_data + '_analysis_position_' + cond_epithet[condition[0]], H_cond)     
        
        np.save(file_location_concatenated_data + '_analysis_behavioural_cluster_' + cond_epithet[condition[0]] + model_name_tag, in_behavioural_cluster_cumulative_cond)  
        np.save(file_location_concatenated_data + '_analysis_monotony_' + cond_epithet[condition[0]] + model_name_tag, self_transition_filtered_cond)          
        np.save(file_location_concatenated_data + '_analysis_settings' + model_name_tag,[plot_colors,cluster_names_reorder])
        
    if not behaviour_only:
        speed_filtered_mean = np.nanmean(speed_filtered_cond/10*40,axis=1)
        speed_filtered_mean = speed_filtered_mean[:len(max_len_frames_all)]
        speed_filtered_sem = np.nanstd(speed_filtered_cond/10*40,axis=1) / np.sqrt(speed_filtered_cond.shape[1])
        speed_filtered_sem = speed_filtered_sem[:len(max_len_frames_all)]
    
        #plot speed
        plt.figure('speed', figsize=figure_size)
        plt.plot(max_len_frames_all*minute_multiplier,speed_filtered_mean,linewidth = 4,color=condition_colors[condition[0]])
        plt.fill_between(max_len_frames_all*minute_multiplier, speed_filtered_mean-speed_filtered_sem, speed_filtered_mean+speed_filtered_sem, color=condition_colors[condition[0]], alpha = .4)
        plt.xlim([0,130])
        plt.pause(.01)
    
    
        angular_speed_filtered_mean = np.nanmean(angular_speed_filtered_cond*40/360,axis=1)
        angular_speed_filtered_mean = angular_speed_filtered_mean[:len(max_len_frames_all)]
        angular_speed_filtered_sem = np.nanstd(angular_speed_filtered_cond*40/360,axis=1) / np.sqrt(angular_speed_filtered_cond.shape[1])
        angular_speed_filtered_sem = angular_speed_filtered_sem[:len(max_len_frames_all)]            
                
        #plot angular speed
        plt.figure('angular speed', figsize=figure_size)
        plt.plot(max_len_frames_all*minute_multiplier,angular_speed_filtered_mean,linewidth = 4,color=condition_colors[condition[0]])
        plt.fill_between(max_len_frames_all*minute_multiplier,angular_speed_filtered_mean-angular_speed_filtered_sem, angular_speed_filtered_mean+angular_speed_filtered_sem, color=condition_colors[condition[0]], alpha = .4)    
        plt.xlim([0,130])
        plt.pause(.01)
        
        in_shelter_filtered_mean = np.nanmean(in_shelter_filtered_cond,axis=1)
        in_shelter_filtered_mean = in_shelter_filtered_mean[:len(max_len_frames_all)]
        in_shelter_filtered_sem = np.nanstd(in_shelter_filtered_cond,axis=1) / np.sqrt(in_shelter_filtered_cond.shape[1])
        in_shelter_filtered_sem = in_shelter_filtered_sem[:len(max_len_frames_all)]
        
        #plot time spent in shelter
        plt.figure('shelter', figsize=figure_size)
        plt.plot(max_len_frames_all*minute_multiplier,in_shelter_filtered_mean,linewidth = 4,color=condition_colors[condition[0]])
        plt.fill_between(max_len_frames_all*minute_multiplier,in_shelter_filtered_mean-in_shelter_filtered_sem, in_shelter_filtered_mean+in_shelter_filtered_sem, color=condition_colors[condition[0]], alpha = .4)    
        plt.xlim([0,130])
        plt.pause(.01)
        
        distance_from_centre_filtered_mean = np.nanmean(distance_from_centre_filtered_cond / 10,axis=1)
        distance_from_centre_filtered_mean = distance_from_centre_filtered_mean[:len(max_len_frames_all)]
        distance_from_centre_filtered_sem = np.nanstd(distance_from_centre_filtered_cond / 10,axis=1) / np.sqrt(speed_filtered_cond.shape[1])
        distance_from_centre_filtered_sem = distance_from_centre_filtered_sem[:len(max_len_frames_all)]
    
        #plot distance from centre      
        plt.figure('distance from centre', figsize=figure_size)
        plt.plot(max_len_frames_all*minute_multiplier,distance_from_centre_filtered_mean,linewidth = 4,color=condition_colors[condition[0]])
        plt.fill_between(max_len_frames_all*minute_multiplier,distance_from_centre_filtered_mean-distance_from_centre_filtered_sem, distance_from_centre_filtered_mean+distance_from_centre_filtered_sem, color=condition_colors[condition[0]], alpha = .4)    
        plt.xlim([0,130])
        plt.pause(.01)
        
        #plot position head mat for each condition  
        plt.figure('position heat map ' + conditions[condition[0]])
        plt.imshow(np.squeeze(np.mean(H_cond,axis=2)).T, vmax=.00001)
        plt.title('position heat map ' + conditions[condition[0]])
        plt.pause(.01)
        
        self_transition_filtered_mean = np.nanmean(self_transition_filtered_cond,axis=1)
        self_transition_filtered_mean = self_transition_filtered_mean[:len(max_len_frames_all)]
        self_transition_filtered_sem = np.nanstd(self_transition_filtered_cond,axis=1) / np.sqrt(self_transition_filtered_cond.shape[1])
        self_transition_filtered_sem = self_transition_filtered_sem[:len(max_len_frames_all)]
    
        #plot distance from centre      
        plt.figure('self-transitions', figsize=figure_size)
        plt.plot(max_len_frames_all*minute_multiplier,self_transition_filtered_mean,linewidth = 4,color=condition_colors[condition[0]])
        plt.fill_between(max_len_frames_all*minute_multiplier,self_transition_filtered_mean-self_transition_filtered_sem, self_transition_filtered_mean+self_transition_filtered_sem, color=condition_colors[condition[0]], alpha = .4)    
        plt.xlim([0,130])
        plt.pause(.01)
        
    #plot position head mat for each condition  
    plt.figure('position heat map ' + conditions[condition[0]])
    plt.imshow(np.squeeze(np.mean(H_cond,axis=2)).T, vmax=.00003*heat_map_bins/50)
    plt.title('position heat map ' + conditions[condition[0]])
    plt.pause(.01)
        
        
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

    
    
    
    

    
