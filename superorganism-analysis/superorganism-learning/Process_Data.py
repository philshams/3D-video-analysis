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
save_folder_location = "C:\\Drive\\Video Analysis\\data\\3D_pipeline\\"
session_name_tags = ['session1']




#  set save and display options
show_images = True
frame_rate_online_display = 1000
do_not_overwrite = True
twoD = True



# ---------------------------
# Select analysis parameters
# ---------------------------
# for initial processing (straighten_upside_down_frames = False)
erode = True; normalize = True
erode_iterations = 4  # for erosion: how (note this will also realign orientation angle which can be turned off in the code)
background_luminance_threshold = 70  # for 2D erode: lower threshold means discarding more pixels


























''' -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------                    Set up processing & wavelet transform                       ------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''


# -------------------------------
# Initialize processing settings
# -------------------------------
level = 5  # how many different spatial scales to use in wavelet transform
discard_scale = 4  # 4 discards 4/5; 6 keeps all
stop_frame = np.inf
if twoD:
    twoD_suffix = '2D'
else:
    twoD_suffix = ''
    normalize = False



# ------------------------------------
# Loop over the sessions listed above
# ------------------------------------
for session in enumerate(session_name_tags):
    print(session[1])
    file_location_saved_data = save_folder_location + session[1] + '\\' + session[1]


    # ---------------------------------------
    # open the and initialize the data video
    # ---------------------------------------
    vid = cv2.VideoCapture(file_location_saved_data + '_data' + twoD_suffix + '.avi'); num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)); end_frame = int(np.min([stop_frame, num_frames]))
    if num_frames < 1:
        raise Exception(file_location_saved_data + '_data' + twoD_suffix + '.avi' + ' not found')

    # ------------------------------------------------------------------------------------------------
    # For straighten upside-down frames, fit the LDA to these data, to find upside-down frames to flip
    # ------------------------------------------------------------------------------------------------
    flip_ind = []
    flip_counter = 0

    # ---------------------------------------------------
    # open current additional data to concatenate / flip
    # ---------------------------------------------------
    # load and initialize data to save
    together_cur = np.load(file_location_saved_data + '_together.npy')             
    start_end_frame_cur =  np.load(file_location_saved_data + '_start_end_frame.npy')  
    start_frame_cur = int(start_end_frame_cur[0])
    end_frame_cur = int(start_end_frame_cur[1])
    all_frames_cur = np.arange(start_frame_cur, end_frame_cur)
    
    frames_of_preprocessed_data_cur = np.load(file_location_saved_data + '_frames.npy') .astype(int) - start_frame_cur

    velocity_cur = np.load(file_location_saved_data + '_velocity.npy')
    coordinates_cur = np.load(file_location_saved_data + '_coordinates.npy')
    orientation_angles_cur = np.angle(velocity_cur[:, 2] + velocity_cur[:, 3] * 1j, deg=True)
    
    position_orientation_velocity_cur = np.ones((together_cur.shape[0], 8)) * np.nan
    wavelet_array = np.ones((39, 39, len(frames_of_preprocessed_data_cur))).astype(np.float16) * np.nan


    ''' -------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------                    Do processing & wavelet transform                       ------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------'''

    # ------------------------------------------------------------------------------------
    # if doing straighten_upside_down_frames, continue while there are yet frames to flip
    # ------------------------------------------------------------------------------------
    while True:

        # --------------
        # read the frame
        # --------------
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_from_preprocessed_data = frame[:, :, 0]
            frame_of_preprocessed_data = int(vid.get(cv2.CAP_PROP_POS_FRAMES)-1)

            if width != 150: # resize
                frame_from_preprocessed_data = cv2.resize(frame_from_preprocessed_data, (150, 150))
                frame = frame_from_preprocessed_data

            # ----------------------
            # erode away background
            # ----------------------
            if erode:  # erode away background
                frame = cv2.erode(frame, np.ones((2, 2), np.uint8), iterations=erode_iterations)
                if twoD:
                    frame[frame > background_luminance_threshold] = 0
            mouse_size = np.sum(frame > 0)

            # ------------------------------------------
            # make background white and normalize image
            # ------------------------------------------
            
            frame[frame == 0] = 250
            frame = 255-cv2.erode(255-frame, np.ones((2, 2), np.uint8), iterations=erode_iterations)
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
            wavelet_array[:, :, frame_of_preprocessed_data] = coeff_array


            # --------------
            # Display images
            # --------------
            if show_images:
                wavelet_recon = pywt.waverec2(coeffs_lowpass, wavelet='db1').astype(np.uint8)
                cv2.imshow('normal image', frame_from_preprocessed_data)
                cv2.imshow('wavelet reconstruction', wavelet_recon)
                cv2.imshow('processed image', frame)

                # ------------------------------------------------------
                # Stop video when finished and notify every 1000 frames
                # ------------------------------------------------------
                if cv2.waitKey(int(1000 / frame_rate_online_display)) & 0xFF == ord('q'):
                    break
            if frame_of_preprocessed_data % 10000 == 0:
                print(str(frame_of_preprocessed_data) + ' out of ' + str(num_frames) + ' frames complete')
            if vid.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break

        else:
            print('data not recognized')
            cv2.waitKey(500)

    ''' -------------------------------------------------------------------------------------------------------------------------------------
    # -----------------------                    Save processing & wavelet transform                       ------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------'''

    # -------------
    # Save wavelets
    # -------------
    vid.release()
    save_file = file_location_saved_data + '_wavelet' + twoD_suffix + '.npy'
    if os.path.isfile(save_file) and do_not_overwrite:
        raise Exception('File already exists')
    if sum(np.isnan(wavelet_array[0,0,:])) > 0:
        raise Exception('NaN remaining in wavelet array')
    print(str(wavelet_array.shape[2]) + ' valid frames saved')
    np.save(save_file, wavelet_array)
    np.save(save_folder_location + 'wavelet_slices', coeff_slices)

    # ------------------------------------------
    # save concatenated / flipped velocity, etc.
    # ------------------------------------------
    position_orientation_velocity_cur[:, 0] = all_frames_cur
    position_orientation_velocity_cur[:, 1] = together_cur
    position_orientation_velocity_cur[frames_of_preprocessed_data_cur, 2:4] = velocity_cur[:, 0:2]
    position_orientation_velocity_cur[frames_of_preprocessed_data_cur, 4] = orientation_angles_cur
    position_orientation_velocity_cur[frames_of_preprocessed_data_cur, 5:7] = coordinates_cur
    
    if os.path.isfile(file_location_saved_data + '_position_orientation_velocity.npy') and do_not_overwrite:
        raise Exception('File already exists')
    np.save(file_location_saved_data + '_position_orientation_velocity.npy', position_orientation_velocity_cur)
    
