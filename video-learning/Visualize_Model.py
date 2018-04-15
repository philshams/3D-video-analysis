'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Visualize the Poses Determined by the Model                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np;
import cv2;
from matplotlib import pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import os
from learning_funcs import reconstruct_from_wavelet, make_color_array, get_trajectory_indices;
from sklearn.externals import joblib
from learning_funcs import set_up_PC_cluster_plot, create_legend

''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
# Select model directory
save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"
data_library_name_tag = 'streamlined'



# Select session directory and video to display
session_name_tag = 'loom_1'
session_video_folder = 'C:\\Drive\\Video Analysis\\data\\baseline_analysis\\27.02.2018\\205_1a\\'
session_video = session_video_folder + 'Chronic_Mantis_stim-default-996386-video-0.avi'
video_num = 0




# --------------------------------
# Select visualization parameters 
# --------------------------------
model_name_tag = '6PC'
model_sequence = False



only_see = True; only_see_component = 4

start_frame = 4500; stop_frame = 47000 
frame_rate = 1000

show_3D_PC_trajectory = False; show_all_3D_PC_data_at_once = False
show_moving_features_plot = False


















''' -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------                            Load data, models, and video                    -----------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''

# ----------
# Find data
# ----------
folder_location_data_library = save_folder_location + data_library_name_tag + '\\'
file_location_data_library = folder_location_data_library + data_library_name_tag
file_location_data_cur = save_folder_location + session_name_tag + '\\' + session_name_tag

# open the video for that session
mouse_vid = cv2.VideoCapture(session_video)


# ----------------
# load model data
# ----------------
model_type = 'hmm'
if model_sequence:  # modify saved file names depending on type of clustering:
    model_type_and_name_tag = 'seq_' + model_name_tag
else:
    model_type_and_name_tag = model_name_tag
    
file_location_data_cur = save_folder_location + session_name_tag + '\\' + session_name_tag

data_for_model_normalized = np.load(file_location_data_cur+ '_data_for_' + model_type + '_normalized.npy')
chosen_components = np.load(file_location_data_cur+ '_chosen_components_' + model_type_and_name_tag + '.npy')
components_binary = np.load(file_location_data_cur+ '_components_binary_' + model_type_and_name_tag + '.npy')
unchosen_components_binary = np.load(file_location_data_cur+ '_unchosen_components_binary_' + model_type_and_name_tag + '.npy')
probabilities = np.load(file_location_data_cur+ '_probabilities_' + model_type_and_name_tag + '.npy')
unchosen_probabilities = np.load(file_location_data_cur+ '_unchosen_probabilities_' + model_type_and_name_tag + '.npy')

num_clusters = probabilities.shape[1]
model = joblib.load(file_location_data_library + '_' + model_type + '_' + model_type_and_name_tag)  # load model

add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at, feature_max, _ = np.load(
    file_location_data_library + '_' + model_type + '_settings_' + model_type_and_name_tag + '.npy')
add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at = np.array(
    [add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at]).astype(int)

# load transform info for reconstruction
pca = joblib.load(file_location_data_library + '_pca')
relevant_wavelet_features = np.load(file_location_data_library + '_relevant_wavelet_features_PCA.npy')
coeff_slices = np.load(save_folder_location + 'wavelet_slices.npy')
level = 5 ; discard_scale = 4  # these must be parameters taken from original wavelet transform

# load position-orientation-velocity for session and out-of-bounds info
if os.path.isfile(file_location_data_cur + '_position_orientation_velocity_corrected.npy'):
    position_orientation_velocity = np.load(file_location_data_cur + '_position_orientation_velocity_corrected.npy')
else:
    position_orientation_velocity = np.load(file_location_data_cur + '_position_orientation_velocity.npy')
    print('loaded non-flip-corrected data')
video_index = position_orientation_velocity[:, 7] == video_num
out_of_bounds = position_orientation_velocity[video_index, 1]
frames = position_orientation_velocity[video_index, 0]
frames = frames[out_of_bounds == 0]

# show a picture of a shelter when the mouse enters the shelter
try:
    shelter = cv2.imread('C:\\Drive\\Video Analysis\\data\\calibration_images\\shelter\\shelter.png')
    shelter = cv2.resize(shelter, (450, 450))
except:
    shelter = np.zeros((450,450)).astype(np.uint8)
    print('shelter image not found')

''' -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------                            Plot Mean Poses                                  -----------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''

# ---------------------------------
# Get mean and variance of clusters
# ---------------------------------
mean_features_model = model.means_
if model_type == 'hmm':
    var_features_model = model.covars_
elif model_type == 'gmm':
    var_features_model = model.covariances_

# ------------------------------------------------
# Display mean trajectory or pose for each cluster
# ------------------------------------------------
# Set up colors
colors = [[0, 0, 255], [169, 118, 14], [10, 205, 10], [160, 0, 120], [0, 100, 250], [170, 120, 220], [0, 140, 140], [100, 100, 100]] * 3
plot_colors = ['red', 'deepskyblue', 'green', 'blueviolet', 'orange', 'lightpink', 'yellow', 'white']
trajectory_pose_size = 300
color_array = make_color_array(colors, trajectory_pose_size)

# Get the number of features used in model
features_used = int(mean_features_model.shape[1] / (2 * windows_to_look_at + 1))
trajectory_position = get_trajectory_indices(windows_to_look_at)



# Plot the mean trajectory or pose:
for n in range(num_clusters):
    trajectory = np.zeros((trajectory_pose_size, trajectory_pose_size * (2 * windows_to_look_at + 1), 3)).astype(
        np.uint8)  # initialize mean trajectory image
    for t in range(2 * windows_to_look_at + 1):
        # Get the mean features for that pose
        mean_features = mean_features_model[n, t * features_used:(t + 1) * features_used]

        # Reconstruct wavelet-transformed data from the PCs
        mean_wavelet_relevant_features = pca.inverse_transform(
            np.append(mean_features[0:num_PCs_used], np.zeros(pca.n_components_ - num_PCs_used)))
        mean_wavelet = np.zeros(39 * 39)
        mean_wavelet[relevant_wavelet_features] = mean_wavelet_relevant_features
        mean_wavelet_array = np.reshape(mean_wavelet, (39, 39))

        # Reconstruct image in pixel space from wavelet-transformed reconstruction
        reconstruction_from_wavelet = reconstruct_from_wavelet(mean_wavelet_array, coeff_slices, level, discard_scale)
        reconstruction_from_wavelet[(reconstruction_from_wavelet<=0) + (reconstruction_from_wavelet>=250)] = 250
        reconstruction_image = cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(trajectory_pose_size, trajectory_pose_size))
        reconstruction_image = cv2.cvtColor(reconstruction_image, cv2.COLOR_GRAY2BGR)
        reconstruction_image = (reconstruction_image * np.squeeze(color_array[:, :, :, n])).astype(np.uint8)

        # rotate image
        M = cv2.getRotationMatrix2D((int(trajectory_pose_size / 2), int(trajectory_pose_size / 2)), -90, 1)
        reconstruction_image = cv2.warpAffine(reconstruction_image, M, (trajectory_pose_size, trajectory_pose_size))

        # Also display an arrow indicating the mean velocity of that cluster
        if add_velocity:
            velocity_of_current_epoch = mean_features_model[:, (t + 1) * features_used - 1 - add_turn]
            # if velocity includes negative values for whatever reason, make only positive by subtracting the lowest velocity value
            min_velocity = np.min(mean_features_model[:,
                                  np.arange(features_used - 1 - add_turn, mean_features_model.shape[1], features_used)])
            max_velocity = np.max(mean_features_model[:,
                                  np.arange(features_used - 1 - add_turn, mean_features_model.shape[1], features_used)])
            if min_velocity < 0:
                velocity_of_current_epoch = velocity_of_current_epoch - min_velocity
                max_velocity = max_velocity - min_velocity
            arrow_speed = velocity_of_current_epoch[n] / max_velocity

            if add_turn:
                turn_of_current_epoch = mean_features_model[:, (t + 1) * features_used - 1]
                # if velocity includes negative values for whatever reason, make only positive by subtracting the lowest velocity value
                min_turn = np.min(
                    mean_features_model[:, np.arange(features_used - 1, mean_features_model.shape[1], features_used)])
                max_turn = np.max(
                    mean_features_model[:, np.arange(features_used - 1, mean_features_model.shape[1], features_used)])
                if min_turn < 0:
                    turn_of_current_epoch = turn_of_current_epoch - min_turn
                    max_turn = max_turn - min_turn
                arrow_turn = turn_of_current_epoch[n] / max_turn


            else:
                arrow_turn = 0
            cv2.arrowedLine(reconstruction_image, (10, trajectory_pose_size - 10),
                            (10 + int(50 * arrow_speed), trajectory_pose_size - 10 - int(20 * arrow_turn)),
                            (0, 0, 0), thickness=3)
        #
        # Display mean pose
        title = 'trajectory ' + str(n + 1)
        trajectory[:, trajectory_pose_size * (t):trajectory_pose_size * (t + 1), :] = reconstruction_image
        cv2.imshow(title, trajectory)
        if cv2.waitKey(int(100)) & 0xFF == ord('q'):
            break

''' -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------                  Plot PCs, Clusters, and Video over time                    -----------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''


# -----------------------------------------------------
# Set up normalized and data videos, and component data
# -----------------------------------------------------
num_frames = int(mouse_vid.get(cv2.CAP_PROP_FRAME_COUNT))
mouse_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
wavelet_array = np.load(save_folder_location + session_name_tag + '\\' + session_name_tag + '_wavelet_corrected.npy')


# -------------------------
# Set up feature-space plot
# -------------------------
plt.close('all')
trajectory_pose_size = 400
show_unchosen_cluster = .15  # threshold to show 2nd-choice cluster
num_PCs_shown = 3

if show_3D_PC_trajectory:
    fig = plt.figure('PC trajectory', figsize=(10, 10))
    ax_3D = fig.add_subplot(111, projection='3d')
    mean_features_model_normalized = mean_features_model / feature_max

    for n in range(num_clusters):
        if show_all_3D_PC_data_at_once:
            component_frames = np.where(components_binary[:, n])[0]
            ax_3D.scatter(data_for_model_normalized[component_frames[::100], 0],
                          data_for_model_normalized[component_frames[::100], 1],
                          data_for_model_normalized[component_frames[::100], 2], color=plot_colors[n], alpha=.6)
            
        ax_3D.scatter(mean_features_model_normalized[n, 0], mean_features_model_normalized[n, 1],
                      mean_features_model_normalized[n, 2], color=plot_colors[n], s=5000, alpha=.3)
        plt.pause(.1)
    ax_3D.set_xlabel('PC1');
    ax_3D.set_ylabel('PC2');
    ax_3D.set_zlabel('PC3')
    ax_3D.set_title('Clusters in PC space')

# ----------------------------
# Set up moving features plot
# ----------------------------
# create coloring array
color_array = make_color_array(colors, 450)

if show_moving_features_plot:
    rolling_fig = set_up_PC_cluster_plot(figsize=(20, 7), xlim=[1500, 2500])
    ax_2D = rolling_fig.add_subplot(111)

    # Now plot the raster of components of the chosen model (pose or sequence)
    for n in range(num_clusters):
        component_frames = np.where(components_binary[:, n])[0]
        ax_2D.scatter(component_frames, np.ones(len(component_frames)) * 1, color=plot_colors[n], alpha=.7, marker='|',s=700)
        unchosen_component_frames = np.where(unchosen_components_binary[:, n] * (unchosen_probabilities > show_unchosen_cluster))[0]
        ax_2D.scatter(unchosen_component_frames, np.ones(len(unchosen_component_frames)) * .9, color=plot_colors[n],
                      alpha=.4, marker='|', s=700)

    # plot the desired number of pc coefficients, varying in time
    ax_2D.plot(data_for_model_normalized[:, 0:num_PCs_shown], linewidth=2)

    # plot the velocity
    if add_velocity:
        ax_2D.plot(data_for_model_normalized[:, -1 - add_turn], color='k', linewidth=2)  # plot speed
        if add_turn:  # plot turn speed or, if available, pose change
            ax_2D.plot(data_for_model_normalized[:, -1], color='gray', linestyle='--', linewidth=2)

    # Create legend
    legend_entries = create_legend(num_PCs_shown, add_velocity, True, False, add_turn)

    # Create line indicating which frame we're looking at
    center_line = ax_2D.plot([0, 0], [-2, 2], color='gray', linestyle='--')

# ------------------------------------------------------------
# Show behaviour and 3D videos, with selected pose also shown 
# ------------------------------------------------------------

i = start_frame - window_size * windows_to_look_at
j = min(np.where(frames >= (start_frame - window_size * windows_to_look_at))[0])
while True:

    ret, frame = mouse_vid.read()

    if ret:
        # Grab frame number, which cluster is active, and its corresponding color
        frame_num_mouse = int(mouse_vid.get(cv2.CAP_PROP_POS_FRAMES))
        mouse_frame = frame[:, :, 0]

        # Display data movie
        if out_of_bounds[i] == 0:  # mouse in arena
            chosen_component = chosen_components[j]
            wavelet_cur = wavelet_array[:,:,j]

            if only_see and chosen_component != only_see_component:
                j += 1
                i = frame_num_mouse - window_size * windows_to_look_at
                continue
            chosen_color = colors[chosen_component]

            reconstruction_from_wavelet  = reconstruct_from_wavelet(wavelet_cur,coeff_slices, level, discard_scale)
            reconstruction_from_wavelet[reconstruction_from_wavelet > 255] = 255
            reconstruction_from_wavelet = cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450))
            reconstruction_from_wavelet = cv2.cvtColor(reconstruction_from_wavelet, cv2.COLOR_GRAY2BGR)
            reconstruction_from_wavelet = (reconstruction_from_wavelet * np.squeeze(color_array[:,:,:,chosen_component])).astype(np.uint8)

            # Move the PC / clusters plot to be centered at the current frame
            if show_moving_features_plot:
                center_line.pop(0).remove()
                center_line = ax_2D.plot([j, j], [-2, 2], color='gray', linestyle='--')
                ax_2D.set_xlim([j - 150, j + 150])
                plt.pause(0.01)

            # add to trajectory plot
            if show_3D_PC_trajectory and not show_all_3D_PC_data_at_once:
                ax_3D.scatter(data_for_model_normalized[j, 0], data_for_model_normalized[j, 1],
                              data_for_model_normalized[j, -1], color=plot_colors[chosen_component], s=100, alpha=.5)
                plt.pause(0.01)
            j += 1
        else:
            chosen_color = [0, 0, 0]
            if out_of_bounds[i] ==1:
                reconstruction_from_wavelet = shelter
            if only_see:
                i = frame_num_mouse - window_size * windows_to_look_at
                continue
        i = frame_num_mouse - window_size * windows_to_look_at
        
        
        # Add colored circle and frame number to behaviour image
        mouse_frame = cv2.cvtColor(mouse_frame, cv2.COLOR_GRAY2BGR)
        cv2.circle(mouse_frame, (50, 50), radius=25, color=chosen_color, thickness=50)
        cv2.putText(mouse_frame, str(i), (50, 1000), 0, 1, 255)
        
        # Display images
        cv2.imshow('behaviour', mouse_frame)
        cv2.imshow('data - wavelet reconstruction', reconstruction_from_wavelet)

        # stop video when donw
        if (frame_num_mouse) % 500 == 0:
            print(str(frame_num_mouse) + ' out of ' + str(num_frames) + ' frames complete')
        if mouse_vid.get(cv2.CAP_PROP_POS_FRAMES) >= min(stop_frame, num_frames):
            break
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
            break
    else:
        print('Problem with Video Playback...')
        cv2.waitKey(int(500))

mouse_vid.release()
