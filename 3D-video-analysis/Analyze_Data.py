'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Plot the Results by Session                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; from matplotlib import pyplot as plt; import os; import warnings; 
from analysis_funcs import plot_analysis, plot_analysis_by_condition, plot_behaviour_analysis_by_condition, get_speed, get_angular_speed
from analysis_funcs import plot_behavioural_clusters, get_behavioural_clusters, get_self_transition, get_position_heat_map, get_distance_from_centre, get_proportion_time_in_shelter
warnings.filterwarnings('once')

''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"
session_name_tags = [['loom_1', 'loom_2', 'loom_3', 'loom_4', 'loom_5', 'loom_6'],
                     ['clicks_1', 'clicks_2', 'clicks_3'],
                     ['post_clicks_1', 'post_clicks_2', 'post_clicks_3']]

data_library_name_tag = 'streamlined'
data_analysis_name_tag = 'analysis'
model_name_tag = '4PC'
model_seq = True


conditions = ['normal with loom','clicks with ultrasound', 'post-clicks with ultrasound'] #first 4 characters must differ
condition_colors = ['black','green','red']


save_filtered_data = True
load_filtered_data = False

analyze_behaviour = True
analyze_position = False
analyze_everything_else = False


# ---------------------------
# Select analysis parameters
# ---------------------------
filter_length = 24000
sigma = 12000 #2400 is one minute

cluster_names = ['locomote', 'groom, hunched sniff, rear', 'pause from locomotion', 'investigate', 'outstretch']
order_of_clusters = [0,2,4,3,1]


figure_size = (16,8)
heat_map_bins = 30
max_trial_in_summary_plots=130














''' -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Load and Prepare data                                     --------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''

# ----------------
# Prepare folders
# ----------------
print('preparing data...')
# find model file location
folder_location_data_library = save_folder_location + data_library_name_tag + '\\'
if not os.path.isdir(folder_location_data_library):
    os.makedirs(folder_location_data_library)
file_location_data_library = folder_location_data_library + data_library_name_tag

# find or create analysis file location
folder_location_data_analysis = save_folder_location + data_analysis_name_tag + '\\'
if not os.path.isdir(folder_location_data_analysis):
    os.makedirs(folder_location_data_analysis)
file_location_data_analysis = folder_location_data_analysis + data_analysis_name_tag
print("saving to " + folder_location_data_analysis)

# create a nametag for saving analysis in the various conditions (1st 4 characters)
cond_epithet = []
for cond in enumerate(conditions):
    cond_epithet.append(cond[1][0:4])

# initialize a couple settings
plt.close('all')
plt.style.use('classic')
num_clusters = len(cluster_names)
plot_colors = ['mediumseagreen','blue','purple','red','darkorange']
plot_colors = plot_colors[:num_clusters]

# --------------------
# Loop over conditions
# --------------------
for condition in enumerate(session_name_tags):
    print('')
    print('analyzing ' + conditions[condition[0]])
    print(str(len(condition[1])) + ' sessions found')
    session_name_tags_in_condition_cur = condition[1]

    # ----------------------------------------------------
    # initialize or load array for per-condition analysis
    # ----------------------------------------------------
    if load_filtered_data:
        if analyze_position:
            distance_from_centre_filtered_cond = np.load(file_location_data_analysis + '_analysis_distance_from_centre_' + cond_epithet[condition[0]] + '.npy')
            H_cond = np.load(file_location_data_analysis + '_analysis_position_' + cond_epithet[condition[0]] + '.npy')
        if analyze_behaviour:
            in_behavioural_cluster_cumulative_cond = np.load(file_location_data_analysis + '_analysis_behavioural_cluster_' + cond_epithet[condition[0]] + model_name_tag + '.npy')
            self_transition_filtered_cond = np.load(file_location_data_analysis + '_analysis_monotony_' + cond_epithet[condition[0]] + model_name_tag + '.npy')            
        if analyze_everything_else:
            speed_filtered_cond = np.load(file_location_data_analysis + '_analysis_speed_' + cond_epithet[condition[0]] + '.npy')
            angular_speed_filtered_cond = np.load(file_location_data_analysis + '_analysis_angular_speed_' + cond_epithet[condition[0]] + '.npy')
            in_shelter_filtered_cond = np.load(file_location_data_analysis + '_analysis_in_shelter_' + cond_epithet[condition[0]] + '.npy')
        plot_colors_reorder, cluster_names_reorder = np.load(file_location_data_analysis + '_analysis_settings' + model_name_tag)
        max_len_frames_all = np.arange(np.sum(np.isfinite(np.mean(in_shelter_filtered_cond,axis=1))))
    else:
        max_session_length = 600000
        if analyze_position:
            distance_from_centre_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
            H_cond = np.zeros((heat_map_bins,heat_map_bins,len(session_name_tags_in_condition_cur)))
        if analyze_behaviour:
            in_behavioural_cluster_cumulative_cond = np.ones((max_session_length,num_clusters+1,len(session_name_tags_in_condition_cur))) * np.nan
            self_transition_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
        if analyze_everything_else:
            speed_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
            angular_speed_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
            in_shelter_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
        frames_cond = np.ones((max_session_length,3,len(session_name_tags_in_condition_cur))) * np.nan
        max_len_frames = []; max_len_frames_uncorrupted = []; max_len_frames_all = []

    # --------------------
    # Loop over sessions
    # --------------------
    for session in enumerate(session_name_tags_in_condition_cur):
        print('')
        print(session[1])
        file_location_data_cur = save_folder_location + session[1] + '\\' + session[1]

        # load postition_orientation_velocity
        if os.path.isfile(file_location_data_cur + '_position_orientation_velocity_corrected.npy'):
            position_orientation_velocity = np.load(
                file_location_data_cur + '_position_orientation_velocity_corrected.npy')
        else:
            position_orientation_velocity = np.load(file_location_data_cur + '_position_orientation_velocity.npy')

        # keep track of which indices are in-bounds and valid
        out_of_bounds = position_orientation_velocity[:, 1]
        disruptions = np.ones(len(out_of_bounds)).astype(bool)
        disruptions[1:] = np.not_equal(out_of_bounds[1:], out_of_bounds[:-1])
        disruptions = disruptions[out_of_bounds == 0]

        #get frame numbers of the (in-bounds) trials to analyze
        frames_all = np.arange(position_orientation_velocity.shape[0])
        frames_uncorrupted = frames_all[np.logical_or(out_of_bounds==0, out_of_bounds==1)]
        frames = frames_all[out_of_bounds==0]
        minute_multiplier = 1 / (40*60)






        ''' -------------------------------------------------------------------------------------------------------------------------------------
        #------------------------                    Analyze each session's data                           --------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------'''

        if analyze_everything_else:
            # ------
            # SPEED
            # ------
            print('analyzing speed...')

            if load_filtered_data:
                speed_filtered_full = speed_filtered_cond[:len(frames_all),session[0]]
            else:
                speed_filtered_full, speed_filtered_cond = get_speed(speed_filtered_cond, position_orientation_velocity,
                                                                     out_of_bounds, disruptions, filter_length, sigma, session, frames, frames_all)

            #plot speed
            plot_analysis('speed', frames_all, minute_multiplier, speed_filtered_full / 10 * 40, conditions, condition, condition_colors, figure_size,
                          title = 'Avg Speed over Time', x_label = 'time in session (minutes)', y_label = 'Speed (~cm/s)')


            # --------------
            # ANGULAR SPEED
            # --------------
            print('analyzing angular speed...')
            if load_filtered_data:
                angular_speed_filtered_full = angular_speed_filtered_cond[:len(frames_all),session[0]]
            else:
                angular_speed_filtered_full, angular_speed_filtered_cond = get_angular_speed(angular_speed_filtered_cond, position_orientation_velocity,
                                                                     out_of_bounds, disruptions, filter_length, sigma, session, frames, frames_all)

            plot_analysis('angular speed', frames_all, minute_multiplier, angular_speed_filtered_full* 40 / 360, conditions, condition, condition_colors, figure_size,
                          title='Avg Angular Speed over Time', x_label='time in session (minutes)', y_label='Angular speed (turns/sec)')


            # ----------------
            # TIME IN SHELTER
            # ----------------
            #calculate time spent in shelter
            print('analyzing shelter...')
            if load_filtered_data:
                in_shelter_filtered = in_shelter_filtered_cond[:len(frames_all),session[0]]
            else:
                in_shelter_filtered, in_shelter_filtered_cond = get_proportion_time_in_shelter(in_shelter_filtered_cond, out_of_bounds, filter_length, sigma, session)

            plot_analysis('shelter', frames_all, minute_multiplier, in_shelter_filtered, conditions, condition, condition_colors, figure_size,
                          title='Proportion of Time Spent in Shelter', x_label='time in session (minutes)', y_label='Proportion of Time')


        if analyze_position:
            # --------------------
            # DISTANCE FROM CENTER
            # --------------------
            print('analyzing position...')

            # find the center of the arena and shelter
            if os.path.isfile(file_location_data_cur + '_arena_roi.npy') and os.path.isfile(file_location_data_cur + '_shelter_roi.npy'):
                arena_roi = np.load(file_location_data_cur + '_arena_roi.npy')
                centre = [arena_roi[0] + arena_roi[2] / 2, arena_roi[1] + arena_roi[3] / 2]
                shelter_roi = np.load(file_location_data_cur + '_shelter_roi.npy')
                shelter = [shelter_roi[0] + shelter_roi[2] / 2, shelter_roi[1] + shelter_roi[3] / 2]
            else:
                raise Exception('please go back create arena and shelter rois from preprocessing script')


            # calculate distance from centre of arena
            if load_filtered_data:
                distance_from_centre_filtered_full = distance_from_centre_filtered_cond[:len(frames_all),session[0]]

            else:
                distance_from_centre_filtered_full, distance_from_centre_filtered_cond, mouse_position_unsheltered = \
                    get_distance_from_centre(distance_from_centre_filtered_cond, position_orientation_velocity, out_of_bounds, 
                                             centre, shelter, filter_length, sigma, frames, frames_uncorrupted, frames_all, session, include_shelter = True)

            plot_analysis('distance from centre', frames_all, minute_multiplier, distance_from_centre_filtered_full/10, conditions, condition, condition_colors, figure_size,
                          title='Distance from Centre of Arena', x_label='time in session (minutes)', y_label='Distance (~cm)')
            
            
            # --------------------
            # POSITION HEAT MAP
            # --------------------            
            #generate position heat map
            if load_filtered_data:
                H = np.squeeze(H_cond[:,:,session[0]])
            else:
                H, H_cond = get_position_heat_map(H_cond, mouse_position_unsheltered, centre, shelter, heat_map_bins, session, arena_dilation_factor = 1.3)
                
            plt.figure('position heat map ' + conditions[condition[0]] + ' session ' + str(session[0]+1))
            plt.imshow(H.T, vmax=np.percentile(H,97))
            plt.title('position heat map ' + conditions[condition[0]] + ' session ' + str(session[0]+1))


        if analyze_behaviour:
            # ----------
            # BEHAVIOUR
            # ----------
            print('analyzing behaviour...')
            if load_filtered_data:
                in_behavioural_cluster_cumulative_full = in_behavioural_cluster_cumulative_cond[:len(frames_all),:,session[0]]
            else:
                in_behavioural_cluster_cumulative_full, in_behavioural_cluster_cumulative_cond, components_binary, frames_model, cluster_names_reorder = \
                    get_behavioural_clusters(in_behavioural_cluster_cumulative_cond, session, model_seq, file_location_data_cur, file_location_data_library,
                                             model_name_tag, order_of_clusters, cluster_names, frames, frames_all, num_clusters, filter_length, sigma)
                
            plot_behavioural_clusters(conditions, condition, session, frames_all, minute_multiplier, in_behavioural_cluster_cumulative_full,
                                      num_clusters, figure_size, plot_colors, cluster_names_reorder)
    
            # ----------------
            # SELF-TRANSITION
            # ----------------
            if load_filtered_data:
                self_transition_filtered_full = self_transition_filtered_cond[:len(frames_all),session[0]]
            else:
                self_transition_filtered_full, self_transition_filtered_cond = get_self_transition(self_transition_filtered_cond, components_binary,
                                                                                                   session, frames_all, frames_model, filter_length, sigma)

            plot_analysis('self-transitions', frames_all, minute_multiplier, self_transition_filtered_full, conditions, condition, condition_colors, figure_size,
                          title='Behavioural Monotony', x_label='time in session (minutes)', y_label='Proportion self-transitions')
    
   
    
        # find the length of the longest session
        if len(frames_all) > len(max_len_frames_all) and not load_filtered_data:
            max_len_frames_all = frames_all


    ''' -------------------------------------------------------------------------------------------------------------------------------------
    #------------------------                    Save and analyze each condition's data                           --------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------'''
        
    if save_filtered_data:
        if analyze_position:            
            np.save(file_location_data_analysis + '_analysis_distance_from_centre_' + cond_epithet[condition[0]], distance_from_centre_filtered_cond)
            np.save(file_location_data_analysis + '_analysis_position_' + cond_epithet[condition[0]], H_cond)
        if analyze_behaviour:
            np.save(file_location_data_analysis + '_analysis_behavioural_cluster_' + cond_epithet[condition[0]] + model_name_tag, in_behavioural_cluster_cumulative_cond)
            np.save(file_location_data_analysis + '_analysis_monotony_' + cond_epithet[condition[0]] + model_name_tag, self_transition_filtered_cond)
        if analyze_everything_else:            
            np.save(file_location_data_analysis + '_analysis_speed_' + cond_epithet[condition[0]], speed_filtered_cond)
            np.save(file_location_data_analysis + '_analysis_angular_speed_' + cond_epithet[condition[0]], angular_speed_filtered_cond)
            np.save(file_location_data_analysis + '_analysis_in_shelter_' + cond_epithet[condition[0]], in_shelter_filtered_cond)
        np.save(file_location_data_analysis + '_analysis_settings' + model_name_tag,[plot_colors,cluster_names_reorder])
    
    
    if analyze_position:     
        #plot distance from centre and position heat map for each condition
        plot_analysis_by_condition('distance from centre', distance_from_centre_filtered_cond / 10, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_trial_in_summary_plots)
        
        plt.figure('position heat map ' + conditions[condition[0]])
        plt.imshow(np.squeeze(np.mean(H_cond,axis=2)).T, vmax=.00003*heat_map_bins/50)
        plt.title('position heat map ' + conditions[condition[0]])
             
    if analyze_behaviour:

        plot_analysis_by_condition('self-transitions', self_transition_filtered_cond, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_trial_in_summary_plots)
        plot_behaviour_analysis_by_condition(in_behavioural_cluster_cumulative_cond, conditions, condition,max_len_frames_all, minute_multiplier, plot_colors, num_clusters, cluster_names_reorder, figure_size)
       

    if analyze_everything_else:       
        # plot speed, angular speed, and proportion of time in shelter
        plot_analysis_by_condition('speed', speed_filtered_cond/10*40, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_trial_in_summary_plots)
        plot_analysis_by_condition('angular speed', angular_speed_filtered_cond*40/360, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_trial_in_summary_plots)
        plot_analysis_by_condition('shelter', in_shelter_filtered_cond, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_trial_in_summary_plots)
   

    

    
    








