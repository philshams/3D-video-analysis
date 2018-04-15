'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Plot the Results by Session                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; from matplotlib import pyplot as plt; import os; import warnings; 
from analysis_funcs import plot_analysis, plot_analysis_by_condition, plot_behaviour_analysis_by_condition, get_speed, get_angular_speed, get_acceleration
from analysis_funcs import plot_behavioural_clusters, get_behavioural_clusters, get_transition, get_position_heat_map, plot_transition_probability
warnings.filterwarnings('once')

''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\3D_pipeline\\"

session_name_tags = [['session1']]

data_library_name_tag = 'test'
data_analysis_name_tag = 'analysis'
model_name_tag = '6PC'
model_seq = False
twoD = True

conditions = ['test session','no intruder'] #first 4 characters must differ
condition_colors = ['black','green','red']


save_filtered_data = True
load_filtered_data = False

analyze_behaviour = True
analyze_position = True
analyze_everything_else = True


# ---------------------------
# Select analysis parameters
# ---------------------------
filter_length = 2400
sigma = 1000 #2400 is one minute

cluster_names = ['chasing' ,'in a line, still' , 'mounting' , 'meet at an angle, still','meet at an angle, moving']
order_of_clusters = [3,4,1,2,5]


figure_size = (16,8)
heat_map_bins = 30
frame_rate = 20
max_minutes_in_summary_plots=11














''' -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                   Load and Prepare data                                     -------------------------------------
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
if twoD:
    model_name_tag = model_name_tag + '2D'

    
    
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
order_of_clusters = [x-1 for x in order_of_clusters]    

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
            transition_cond = np.load(file_location_data_analysis + '_transition_probability_' + cond_epithet[condition[0]] + model_name_tag + '.npy')            
        if analyze_everything_else:
            speed_filtered_cond = np.load(file_location_data_analysis + '_analysis_speed_' + cond_epithet[condition[0]] + '.npy')
            acceleration_filtered_cond = np.load(file_location_data_analysis + '_analysis_acceleration_' + cond_epithet[condition[0]] + '.npy')
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
            transition_cond = np.zeros((num_clusters,num_clusters,len(session_name_tags_in_condition_cur)))
        if analyze_everything_else:
            speed_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
            acceleration_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
            angular_speed_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
            in_shelter_filtered_cond = np.ones((max_session_length,len(session_name_tags_in_condition_cur))) * np.nan
        frames_cond = np.ones((max_session_length,3,len(session_name_tags_in_condition_cur))) * np.nan
        max_len_frames = []; max_len_frames_all = []; max_len_frames_all = []

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
        together = position_orientation_velocity[:, 1].astype(bool)
        disruptions = np.ones(len(together)).astype(bool)
        disruptions[1:] = np.not_equal(together[1:], together[:-1])
        disruptions = disruptions[together]

        #get frame numbers of the (in-bounds) trials to analyze
        frames_all = np.arange(position_orientation_velocity.shape[0])
        frames = frames_all[together]
        minute_multiplier = 1 / (frame_rate*60)






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
                                                                     together, disruptions, filter_length, sigma, session, frames, frames_all)

            #plot speed
            plot_analysis('speed', frames_all, minute_multiplier, speed_filtered_full / 10 * frame_rate, conditions, condition, condition_colors, figure_size,
                          title = 'Avg Speed over Time', x_label = 'time in session (minutes)', y_label = 'Speed (~cm/s)')

            # ------------
            # ACCELERATION
            # ------------
            print('analyzing acceleration...')

            if load_filtered_data:
                acceleration_filtered_full = acceleration_filtered_cond[:len(frames_all),session[0]]
            else:
                acceleration_filtered_full, acceleration_filtered_cond = get_acceleration(acceleration_filtered_cond, position_orientation_velocity,
                                                                     together, disruptions, filter_length, sigma, session, frames, frames_all)

            #plot speed
            plot_analysis('acceleration', frames_all, minute_multiplier, acceleration_filtered_full / 10 * frame_rate * frame_rate, conditions, condition, condition_colors, figure_size,
                          title = 'Avg Acceleration over Time', x_label = 'time in session (minutes)', y_label = 'Acceleration magnitude (~cm/s^2)')



            # --------------
            # ANGULAR SPEED
            # --------------
            print('analyzing angular speed...')
            if load_filtered_data:
                angular_speed_filtered_full = angular_speed_filtered_cond[:len(frames_all),session[0]]
            else:
                angular_speed_filtered_full, angular_speed_filtered_cond = get_angular_speed(angular_speed_filtered_cond, position_orientation_velocity,
                                                                     together, disruptions, filter_length, sigma, session, frames, frames_all)

            plot_analysis('angular speed', frames_all, minute_multiplier, angular_speed_filtered_full* frame_rate / 360, conditions, condition, condition_colors, figure_size,
                          title='Avg Angular Speed over Time', x_label='time in session (minutes)', y_label='Angular speed (turns/sec)')




        if analyze_position:           
            # --------------------
            # POSITION HEAT MAP
            # --------------------            
            #generate position heat map
            if load_filtered_data:
                H = np.squeeze(H_cond[:,:,session[0]])
            else:
                H, H_cond = get_position_heat_map(H_cond, position_orientation_velocity, together, heat_map_bins, session)
                
            plt.style.use('classic')
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
    
            # -------------------------
            # TRANSITION PROBABILITIES
            # -------------------------
            if load_filtered_data:
                transition = np.squeeze(transition_cond[:,:,session[0]])
            else:
                transition, transition_cond = get_transition(transition_cond, components_binary, session, file_location_data_cur, num_clusters, model_seq, model_name_tag, frames_all, frames_model, filter_length, sigma)

            plot_transition_probability(transition, False, conditions, condition, session, .06, frame_rate, cluster_names, num_clusters)
    
        # find the length of the longest session
        if len(frames_all) > len(max_len_frames_all) and not load_filtered_data:
            max_len_frames_all = frames_all


    ''' -------------------------------------------------------------------------------------------------------------------------------------
    #------------------------                    Save and analyze each condition's data                           -----------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------'''
        
    if save_filtered_data:
        if analyze_position:            
            np.save(file_location_data_analysis + '_analysis_position_' + cond_epithet[condition[0]], H_cond)
        if analyze_behaviour:
            np.save(file_location_data_analysis + '_analysis_behavioural_cluster_' + cond_epithet[condition[0]] + model_name_tag, in_behavioural_cluster_cumulative_cond)
            np.save(file_location_data_analysis + '_transition_probability_' + cond_epithet[condition[0]] + model_name_tag + '.npy', transition_cond)            
            np.save(file_location_data_analysis + '_analysis_settings' + model_name_tag,[plot_colors,cluster_names_reorder])
        if analyze_everything_else:            
            np.save(file_location_data_analysis + '_analysis_speed_' + cond_epithet[condition[0]], speed_filtered_cond)
            np.save(file_location_data_analysis + '_analysis_acceleration_' + cond_epithet[condition[0]], speed_filtered_cond)
            np.save(file_location_data_analysis + '_analysis_angular_speed_' + cond_epithet[condition[0]], angular_speed_filtered_cond)

        
    
    
    if analyze_position:     
        #plot distance from centre and position heat map for each condition
        plt.style.use('classic')
        plt.figure('position heat map ' + conditions[condition[0]])
        plt.imshow(np.squeeze(np.mean(H_cond,axis=2)).T, vmax=.0001*heat_map_bins/50)
        plt.title('position heat map ' + conditions[condition[0]])
             
    if analyze_behaviour:
        plot_behaviour_analysis_by_condition(in_behavioural_cluster_cumulative_cond, conditions, condition,max_len_frames_all, minute_multiplier, plot_colors, num_clusters, cluster_names_reorder, figure_size)
        plot_transition_probability(np.mean(transition_cond,axis=2), True, conditions, condition, session, .05, frame_rate, cluster_names, num_clusters)                    

    if analyze_everything_else:       
        # plot speed, angular speed, and proportion of time in shelter
        plot_analysis_by_condition('speed', speed_filtered_cond/10*frame_rate, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_minutes_in_summary_plots)
        plot_analysis_by_condition('acceleration', acceleration_filtered_cond/10*frame_rate*frame_rate, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_minutes_in_summary_plots)
        plot_analysis_by_condition('angular speed', angular_speed_filtered_cond*frame_rate/360, max_len_frames_all, frames_all, minute_multiplier, condition_colors, condition, figure_size, max_minutes_in_summary_plots)

    
    








