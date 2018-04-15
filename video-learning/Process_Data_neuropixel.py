'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Straighten, Remove Background, and Perform Wavelet Transform on video        ---------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np;
import pywt;
import cv2;
import os;
import glob;
from learning_funcs import get_biggest_contour, filter_features;
from sklearn.externals import joblib

''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\neuropixel\\"
session_name_tags = ['neuropixel_1']
data_library_name_tag = 'test'



#  set save and display options
show_images = True
frame_rate_online_display = 40
straighten_upside_down_frames = True  # use existing LDA and wavelet transform to straighten upside-down frames
do_not_overwrite = False





# ---------------------------
# Select analysis parameters
# ---------------------------
# for initial processing (straighten_upside_down_frames = False)
erode = True; normalize = True; reject = True; ellipse_mask = True
dilate_iterations = 4  # for erosion: how (note this will also realign orientation angle which can be turned off in the code)
mouse_size_pixels = [1000, 5000]  # for reject: mouse must be within these sizes, in number of pixels
background_luminance_threshold = 50  # for erode: lower threshold means discarding more pixels
background_luminance_threshold_2 = 85
width_threshold = 1.3  # ellipticality of ellipse needed for orientation angle changes to be registered

# for flip correction (straighten_upside_down_frames = True)
flip_threshold = .95; flip_threshold_filtered = .9
























''' -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------                    Set up processing & wavelet transform                       ------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''


# -------------------------------
# Initialize processing settings
# -------------------------------
if straighten_upside_down_frames:
    upside_down = False
    reject = False
    file_location_data_library = save_folder_location + data_library_name_tag + '\\' + data_library_name_tag
    lda = joblib.load(file_location_data_library + '_lda')
    relevant_ind = np.load(file_location_data_library + '_relevant_wavelet_features_LDA.npy')
    velocity_scaler = joblib.load(file_location_data_library + '_lda_velocity_scaling') 
    scaling_array = np.load(file_location_data_library + '_lda_scaling.npy')
    mean_vel, std_vel, wavelet_feature_means, wavelet_feature_stds = scaling_array[0][0], scaling_array[1][0], scaling_array[2], scaling_array[3]
else:
    if erode:
        realign = True
        recenter = True
    else:
        realign = False
        recenter = False
    upside_down = True # flip video and save as a separate version, for later analysis
level = 5  # how many different spatial scales to use in wavelet transform
discard_scale = 4  # 4 discards 4/5; 6 keeps all
stop_frame = np.inf


# ------------------------------------
# Loop over the sessions listed above
# ------------------------------------
for session in enumerate(session_name_tags):
    print(session[1])
    file_location_saved_data = save_folder_location + session[1] + '\\' + session[1]


    # ---------------------------------------
    # open the and initialize the data video
    # ---------------------------------------
    vid = cv2.VideoCapture(file_location_saved_data + '_data.avi'); num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)); end_frame = int(np.min([stop_frame, num_frames]))
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if num_frames < 1:
        raise Exception(file_location_saved_data + '_data.avi' + ' not opening')

    # ------------------------------------------------------------------------------------------------
    # For straighten upside-down frames, fit the LDA to these data, to find upside-down frames to flip
    # ------------------------------------------------------------------------------------------------
    flip_ind = []
    flip_counter = 0
    if straighten_upside_down_frames:  # use the LDA to flip flipped frames
        # load wavelet transformed data
        wavelet_array = np.load(file_location_saved_data + '_wavelet.npy')
        wavelet_array = np.reshape(wavelet_array, (39 * 39, wavelet_array.shape[2]))
        wavelet_array_relevant_features = wavelet_array[relevant_ind, :].T

        # get velocity, body orientation, and disruptions where the velocity calculation is n/a
        position_orientation_velocity_cur = np.load(file_location_saved_data + '_position_orientation_velocity.npy')
        out_of_bounds_cur = position_orientation_velocity_cur[:,1]
        mouse_in_bounds_index = position_orientation_velocity_cur[:, 1] == 0
        velocity_toward_head_dir = position_orientation_velocity_cur[mouse_in_bounds_index,2:3]
        orientation_angle = position_orientation_velocity_cur[mouse_in_bounds_index, 4]
        disruptions = position_orientation_velocity_cur[mouse_in_bounds_index, 7]

        # rescale data
        velocity_toward_head_dir[disruptions == 1, :] = 0  # remove spurious velocities

        # don't flip frames where the mouse is moving quickly forward
        slow_forward_velocity_ind = np.squeeze(velocity_toward_head_dir <= (np.median(velocity_toward_head_dir) + std_vel))
        print('only flip frames below ' + str(np.median(velocity_toward_head_dir) + std_vel) + ' pixels/frame')

        # saturate velocity above spurious level
        velocity_toward_head_dir[(velocity_toward_head_dir[:,0] - mean_vel) > (6 * std_vel), 0] = mean_vel + 6 * std_vel
        velocity_toward_head_dir[(velocity_toward_head_dir[:,0] - mean_vel) < (-6 * std_vel), 0] = mean_vel - 6 * std_vel

        # rescale data using same scalers as used in the LDA
        velocity_toward_head_dir = velocity_scaler.transform(velocity_toward_head_dir[:, 0:1])
        wavelet_array_relevant_features = (wavelet_array_relevant_features - wavelet_feature_means) / wavelet_feature_stds

        # append data and velocity
        features = np.concatenate((wavelet_array_relevant_features, velocity_toward_head_dir), axis=1)

        # fit lda to the features to classify as straight or upside-down
        proportion_straight = lda.score(features, np.ones(features.shape[0]))
        print(str(int(100 * (1 - proportion_straight))) + '% of trials classified as upside-down')

        # calculate probabilities/predictions for a particular session
        predicted_prob_up = lda.predict_proba(features)
        predicted_state_flip = (predicted_prob_up[:, 0] > flip_threshold) * slow_forward_velocity_ind

        # smooth flip decision for continuity
        predicted_state_flip_filtered = np.zeros((len(predicted_state_flip), 1))
        predicted_state_flip_filtered = (filter_features(predicted_prob_up, 3, np.inf) > flip_threshold_filtered)[:, 0]

        # flip frames over flip threshold and intermingled frames
        flip_ind = np.where((predicted_state_flip + predicted_state_flip_filtered) > 0)[0]
#        flip_ind = np.where(predicted_state_flip_filtered)[0]
        print(str(int(100 * len(flip_ind) / len(predicted_state_flip))) + '% of ' + str(
            len(predicted_state_flip)) + ' trials flipped after adjustment')

        # set up indices for the frames to flip, and additional data to flip (velocity, orientation)
        frames_of_processed_data = position_orientation_velocity_cur[mouse_in_bounds_index, 0]
#        frames_of_preprocessed_data_cur = np.load(file_location_saved_data + '_frames.npy')
        frames_of_preprocessed_data_cur = np.where(np.logical_or(out_of_bounds_cur == 0,out_of_bounds_cur == 3))[0].astype(float)
        frames_of_processed_data_to_flip = frames_of_processed_data[flip_ind].astype(int)
        velocity_cur = np.load(file_location_saved_data + '_velocity.npy')
        velocity_flipped = np.ones((len(frames_of_processed_data_to_flip), 2)) * np.nan
        orientation_angles_cur = position_orientation_velocity_cur[np.logical_or(position_orientation_velocity_cur[:, 1] == 0, position_orientation_velocity_cur[:, 1] == 3),4]
        orientation_angles_flipped = np.ones(len(frames_of_processed_data_to_flip)) * np.nan

    # ---------------------------------------------------
    # open current additional data to concatenate / flip
    # ---------------------------------------------------
    else:
        # load and initialize data to save
        out_of_bounds_cur = np.load(file_location_saved_data + '_out_of_bounds.npy')[:,0]            
        vid_nums = np.load(file_location_saved_data + '_out_of_bounds.npy')[:,1]        
        disruptions_cur = np.load(file_location_saved_data + '_disruption.npy').astype(float)
        disruption = 0
        frames_of_preprocessed_data_cur = np.where(np.logical_or(out_of_bounds_cur == 0,out_of_bounds_cur == 3))[0].astype(float)
        frames_of_preprocessed_data_cur_copy = frames_of_preprocessed_data_cur.copy().astype(int)
        first_frame = int(frames_of_preprocessed_data_cur[0])
        velocity_cur = np.load(file_location_saved_data + '_velocity.npy')
        coordinates_cur = np.load(file_location_saved_data + '_coordinates.npy')
        orientation_angles_cur = np.angle(velocity_cur[:, 2] + velocity_cur[:, 3] * 1j, deg=True)
        position_orientation_velocity_cur = np.ones((out_of_bounds_cur.shape[0], 8)) * np.nan
        wavelet_array = np.ones((39, 39, end_frame, upside_down + 1)).astype(np.float16) * np.nan


    ''' -------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------                    Do processing & wavelet transform                       ------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------'''

    # ------------------------------------------------------------------------------------
    # if doing straighten_upside_down_frames, continue while there are yet frames to flip
    # ------------------------------------------------------------------------------------
    while flip_counter < len(flip_ind) or (not straighten_upside_down_frames):
        if straighten_upside_down_frames:
            frame_of_preprocessed_data = np.where(frames_of_preprocessed_data_cur == \
                                                  frames_of_processed_data_to_flip[flip_counter])[0][0]
            frame_of_processed_data = np.where(frames_of_processed_data == frames_of_processed_data_to_flip[flip_counter])[0]
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_of_preprocessed_data)

        # --------------
        # read the frame
        # --------------
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_from_preprocessed_data = frame[:, :, 0]
            frame_of_preprocessed_data = int(vid.get(cv2.CAP_PROP_POS_FRAMES)-1)

            skip_frame = False
            if width != 150: # resize
                frame_from_preprocessed_data = cv2.resize(frame_from_preprocessed_data, (150, 150))

            # ------------------------------------------------------------------------------
            # loop over the upright and inverted versions of the data (if upside_down==True)
            # ------------------------------------------------------------------------------
            for image_flip in range(upside_down + 1):
                if image_flip == 1:  # create inverted version of data
                    M = cv2.getRotationMatrix2D((int(150 / 2), int(150 / 2)), 180, 1)
                    frame = cv2.warpAffine(frame_from_preprocessed_data, M, (150, 150))
                else:
                    frame = frame_from_preprocessed_data

                # --------------------------------------------
                # flip frame before processing, if applicable
                # --------------------------------------------
                if straighten_upside_down_frames:
                    M = cv2.getRotationMatrix2D((int(150 / 2), int(150 / 2)), 180, 1)
                    frame = cv2.warpAffine(frame, M, (150, 150))
                    orientation_angles_flipped[flip_counter] = orientation_angles_cur[frame_of_preprocessed_data]  - 180* (2*(orientation_angles_cur[frame_of_preprocessed_data] > 0) - 1)

                    velocity_flipped[flip_counter, :] = -1 * velocity_cur[frame_of_preprocessed_data, 0:2]

                frame_copy_2 = frame.copy()
                # ----------------------
                # erode away background
                # ----------------------             
                if erode:  # erode away background
#                    frame = cv2.erode(frame, np.ones((2, 2), np.uint8), iterations=int(dilate_iterations/2))
                    frame_copy = frame.copy()
                    frame_copy[frame > background_luminance_threshold] = 0
                    
                    contours, big_cnt_ind, center_x, center_y, _ = get_biggest_contour(frame_copy)
                    blank = np.zeros((150,150)).astype(np.uint8)
                    contour_mask = cv2.drawContours(blank, contours, big_cnt_ind, color=(1, 1, 1), thickness=cv2.FILLED)
                    contour_mask = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=dilate_iterations)
                    contour_mask = cv2.erode(contour_mask, np.ones((2, 2), np.uint8), iterations=int(dilate_iterations/2))
                    
                    frame = frame * contour_mask
                    frame[frame > background_luminance_threshold_2] = 0
                    
                    
#                    frame = cv2.erode(frame, np.ones((2, 2), np.uint8), iterations=3)

                    cv2.imshow('contour',contour_mask*200)
                    
                mouse_size = np.sum(frame > 0)


                # ----------------------------------------------
                # throw out frames with a non-mouse-sized object
                # ----------------------------------------------
                if not straighten_upside_down_frames:

                    if (mouse_size < mouse_size_pixels[0] or mouse_size > mouse_size_pixels[1]) and reject:
                        disruption = 1
                        disruptions_cur[frame_of_preprocessed_data] = np.nan

                        frame_num_whole_data = frames_of_preprocessed_data_cur[frame_of_preprocessed_data].astype(int)
                        frames_of_preprocessed_data_cur[frame_of_preprocessed_data] = np.nan
                        velocity_cur[frame_of_preprocessed_data, :] = np.ones(4) * np.nan
                        coordinates_cur[frame_of_preprocessed_data, :] = np.ones(2) * np.nan
                        orientation_angles_cur[frame_of_preprocessed_data] = np.nan

                        out_of_bounds_cur[frame_num_whole_data] = 3
                        wavelet_array[:, :, frame_of_preprocessed_data] = np.nan
                        skip_frame = True
                        break
                    elif disruption == 1:
                        disruptions_cur[frame_of_preprocessed_data] = 1
                        disuption = 0
                                
                # ---------------
                # recenter mouse
                # ---------------
               
                if recenter:
                    contours, big_cnt_ind, center_x, center_y, _ = get_biggest_contour(frame)                    
                    blank = np.zeros((150,150)).astype(np.uint8)
                    if center_x <75:
                        x_start, x_stop = 0, 2*center_x
                        x_start_new_frame, x_stop_new_frame = 75 - (center_x), 75 + (center_x)
                    else:
                        x_start, x_stop = 2*center_x - 150, 150
                        x_start_new_frame, x_stop_new_frame = 75 - (150-center_x), 75 + (150-center_x)
                    if center_y < 75:
                        y_start, y_stop = 0, 2*center_y
                        y_start_new_frame, y_stop_new_frame = 75 - (center_y), 75 + (center_y)
                    else:
                        y_start, y_stop = 2*center_y - 150, 150
                        y_start_new_frame, y_stop_new_frame = 75 - (150-center_y), 75 + (150-center_y)
                        
                    if ellipse_mask:
                        blank[x_start_new_frame:x_stop_new_frame, y_start_new_frame:y_stop_new_frame] = \
                        frame_copy_2[x_start:x_stop, y_start:y_stop]
                        frame_copy_2 = blank
                    else:
                        blank[x_start_new_frame:x_stop_new_frame, y_start_new_frame:y_stop_new_frame] = \
                        frame[x_start:x_stop, y_start:y_stop]
                        frame = blank

                    
                
                
                # ----------------------------------------------
                # straighten mouse, update angle, save to array
                # ----------------------------------------------
                if realign:
                    contours, big_cnt_ind, center_x, center_y, _ = get_biggest_contour(frame)
                    ellipse = cv2.fitEllipse(contours[big_cnt_ind])
                    ellipse_angle = ellipse[2] - 180 * (ellipse[2] > 90)

                    M = cv2.getRotationMatrix2D((int(150 / 2), int(150 / 2)), ellipse_angle, 1)
                    if ellipse_mask:
                        frame_copy_2 = cv2.warpAffine(frame_copy_2, M, (150, 150))
                    else:
                        frame = cv2.warpAffine(frame, M, (150, 150))
                    ellipse_width = (ellipse[1][1] / ellipse[1][0])

                    if ellipse_width > width_threshold:  # if mouse is circular, use the previous orientation angle instead
                        orientation_angles_cur[frame_of_preprocessed_data] = orientation_angles_cur[frame_of_preprocessed_data] - ellipse_angle
                    elif np.isfinite(orientation_angles_cur[frame_of_preprocessed_data - 1]):
                        orientation_angles_cur[frame_of_preprocessed_data] = orientation_angles_cur[frame_of_preprocessed_data - 1]


                # -------------------
                # apply ellipse mask
                # -------------------
                if ellipse_mask:
                    cv2.imshow('test', frame_copy_2)
                    ellipse_mask_array = np.zeros((150,150)).astype(np.uint8)
                    cv2.ellipse(ellipse_mask_array,(75,75),(30,55),0,0,360,1,thickness=-1)
                    frame = frame_copy_2 * ellipse_mask_array
                    frame[frame > background_luminance_threshold_2] = 0
                    cv2.imshow('test2', frame)

                # ------------------------------------------
                # make background white and normalize image
                # ------------------------------------------
                frame[frame < 10] = 250
                frame = 255-cv2.erode(255-frame, np.ones((2, 2), np.uint8), iterations=2)
                if normalize:  # normalize by the mean pixel value
                    zero_mean_array = frame[frame < 250] / np.mean(frame[frame < 250]) * 127.5
                    zero_mean_array[zero_mean_array >= 250] = 250
                    frame[frame < 250] = (zero_mean_array + (127 - np.mean(zero_mean_array))).astype(np.uint8)

                # -----------------------------
                # extract wavelet coefficients
                # -----------------------------
                coeffs_lowpass = [[], [], [], [], [], []]
                coeffs = pywt.wavedec2(frame, wavelet='db1', level=level)
                # discard coefficients at too coarse of a spaital scale, as set by discard_scale
                for i in range(level + 1):
                    if i < discard_scale:
                        coeffs_lowpass[i] = coeffs[i]
                    else:
                        coeffs_lowpass[i] = [None, None, None]
                # place coefficients in an array, and take coeff_slices index for later reconstruction
                coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs[0:discard_scale])
                if straighten_upside_down_frames:
                    wavelet_array[:, frame_of_processed_data] = np.reshape(coeff_array, (39 * 39, 1))
                    flip_counter += 1
                else:
                    wavelet_array[:, :, frame_of_preprocessed_data, image_flip] = coeff_array


            if skip_frame:
                continue
            # --------------
            # Display images
            # --------------
            if show_images:
                wavelet_recon = pywt.waverec2(coeffs_lowpass, wavelet='db1').astype(np.uint8)
                cv2.imshow('normal image', frame_from_preprocessed_data)
                cv2.imshow('wavelet reconstruction', wavelet_recon)
                if upside_down or straighten_upside_down_frames:
                    cv2.imshow('flipped and processed image', frame)

                # ------------------------------------------------------
                # Stop video when finished and notify every 1000 frames
                # ------------------------------------------------------
                if cv2.waitKey(int(1000 / frame_rate_online_display)) & 0xFF == ord('q'):
                    break
            if frame_of_preprocessed_data % 10000 == 0 and not straighten_upside_down_frames:
                print(str(frame_of_preprocessed_data) + ' out of ' + str(num_frames) + ' frames complete')
            if vid.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break

        else:
            if vid.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break
            print('data not recognized')
            cv2.waitKey(500)

    ''' -------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------                    Save processing & wavelet transform                       ------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------'''

    # -------------
    # Save wavelets
    # -------------
    vid.release()
    if straighten_upside_down_frames:
        if os.path.isfile(file_location_saved_data + '_wavelet_corrected.npy') and do_not_overwrite:
            raise Exception('File already exists')
        wavelet_array = np.reshape(wavelet_array, (39, 39, wavelet_array.shape[1]))
        np.save(file_location_saved_data + '_wavelet_corrected.npy', wavelet_array)
        position_orientation_velocity_cur[frames_of_processed_data_to_flip, 2:4] = velocity_flipped
        position_orientation_velocity_cur[frames_of_processed_data_to_flip, 4] = orientation_angles_flipped

        if os.path.isfile(
                file_location_saved_data + '_position_orientation_velocity_corrected.npy') and do_not_overwrite:
            raise Exception('File already exists')
        np.save(file_location_saved_data + '_position_orientation_velocity_corrected.npy', position_orientation_velocity_cur)

    else:
        save_file = file_location_saved_data + '_wavelet.npy'
        if os.path.isfile(save_file) and do_not_overwrite:
            raise Exception('File already exists')
        non_nan_ind = ~np.isnan(wavelet_array[0, 0, :, 0])

        wavelet_array_up = np.squeeze(wavelet_array[:, :, non_nan_ind, 0])
        print(str(wavelet_array_up.shape[2]) + ' valid frames saved')
        np.save(file_location_saved_data + '_wavelet.npy', wavelet_array_up)
        if upside_down:
            wavelet_array_down = np.squeeze(wavelet_array[:, :, non_nan_ind, 1])
            np.save(file_location_saved_data + '_upside_down_wavelet', wavelet_array_down)
        # ------------------------------------------
        # save concatenated / flipped velocity, etc.
        # ------------------------------------------
        position_orientation_velocity_cur[frames_of_preprocessed_data_cur_copy, 0] = frames_of_preprocessed_data_cur
        position_orientation_velocity_cur[:, 1] = out_of_bounds_cur
        position_orientation_velocity_cur[frames_of_preprocessed_data_cur_copy, 2:4] = velocity_cur[:, 0:2]
        position_orientation_velocity_cur[frames_of_preprocessed_data_cur_copy, 4] = orientation_angles_cur
        position_orientation_velocity_cur[frames_of_preprocessed_data_cur_copy, 5:7] = coordinates_cur
        position_orientation_velocity_cur[:, 7] = vid_nums
        
        if os.path.isfile(file_location_saved_data + '_position_orientation_velocity.npy') and do_not_overwrite:
            raise Exception('File already exists')
        np.save(file_location_saved_data + '_position_orientation_velocity.npy', position_orientation_velocity_cur)
        np.save(save_folder_location + 'wavelet_slices', coeff_slices)
