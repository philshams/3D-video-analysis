'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform Wavelet Transform on 3D mouse video                             --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import pywt; import cv2; import os


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_location = 'C:\Drive\Video Analysis\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'
session_name_tags = ['normal_1_0','normal_1_1','normal_1_2',
                    'normal_2_0','normal_2_1','normal_2_2',
                    'normal_3_0','normal_3_1',
                    'normal_4_0','normal_4_1','normal_4_2',
                    'normal_5_0','normal_5_1',
                    'normal_6_0','normal_6_1','normal_6_2',
                    'clicks_1_0','clicks_2_0','clicks_3_0',
                    'post_clicks_1_0','post_clicks_2_0','post_clicks_3_0']

session_name_tags = ['normal_1_0','normal_1_1','normal_1_2']

upside_down = False #use flipped frames to create the LDA model
straighten_upside_down_frames = False #use existing LDA to straighten upside-down frames
    

# ---------------------------
# Select analysis parameters
# ---------------------------
frame_rate = 60
end_frame = np.inf
show_images = True
save_wavelets = False

save_concatenate_data = False
only_save_concatenate_data = False
concatenated_data_name_tag = 'session_1'
do_not_overwrite = False

level = 5 # how many different spatial scales to use
discard_scale = 4 # 4 discards 4/5; 6 keeps all




#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Do wavelet transform                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ----------------------
# Set up video playback
# ----------------------
if upside_down:
    upside_down_suffix = '_upside_down'
else:
    upside_down_suffix = ''
    
if save_concatenate_data:
    frames = np.array([])
    coordinates = []
    velocity = np.array(([],[],[],[])).T
    disruption = []
    out_of_bounds = []
    
file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag   + '\\' + concatenated_data_name_tag   
        
for v in range(len(session_name_tags)):
    session_name_tag = session_name_tags[v]
    file_location_data = file_location + data_folder + analysis_folder + session_name_tag + '\\' + session_name_tag
    print(file_location_data)
       
    # open current additional data to concatenate / flip
    if save_concatenate_data:
        out_of_bounds_cur = np.load(file_location_data + '_out_of_bounds.npy')
        disruption_cur = np.load(file_location_data + '_disruption.npy')
        frames_cur = np.load(file_location_data + '_frames.npy')
        velocity_cur = np.load(file_location_data + '_velocity.npy')
        coordinates_cur   = np.load(file_location_data + '_coordinates.npy')  

    if not only_save_concatenate_data: #skip wavelet analysis, just concatenate velocity, etc. (without correcting upside-down frames)
        
        # open the and initialize the data video
        vid = cv2.VideoCapture(file_location_data + '_data' + upside_down_suffix + '.avi')
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        stop_frame = int(np.min([end_frame, num_frames]))
        wavelet_array = np.zeros((39,39,stop_frame)).astype(np.float16)
        
        # load LDA model
        
        if straighten_upside_down_frames:
            lda = joblib.load(file_location_concatenated_data + '_lda')
            relevant_ind = np.load(file_location_concatenated_data + '_wavelet_relevant_ind_LDA.npy')

         
    # ---------------------------------------------
    # for each frame, perform wavelet decomposition
    # ---------------------------------------------        
        while True:
            ret, frame = vid.read() # get the frame
            if ret: 
                # grab and resize the frame
                frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
                frame = frame[:,:,0]*2
                if width != 150:
                    frame = cv2.resize(frame,(150,150)) #resize
                
                # -----------------------------
                # extract wavelet coefficients
                # -----------------------------
                for flip in [0,1]:
                    coeffs_lowpass = [[],[],[],[],[],[]]
                    coeffs = pywt.wavedec2(frame, wavelet='db1',level = level)
                    for i in range(level+1):
                        #discard coefficients at too coarse of a spaital scale, as set by discard_scale
                        if i < discard_scale:
                            coeffs_lowpass[i] = coeffs[i]
                        else:
                            coeffs_lowpass[i] = [None,None,None]
                    wavelet_recon = pywt.waverec2(coeffs_lowpass, wavelet='db1').astype(np.uint8)
    
                    #place coefficients in an array, and take coeff_slices index for later reconstruction       
                    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs[0:discard_scale])
                    wavelet_array[:,:,frame_num-1] = coeff_array
                    
                    # --------------------------
                    # check if image is flipped
                    # --------------------------  
                    if straighten_upside_down_frames and flip==0:
                        features = np.resize(coeff_array,(39*39))[relevant_ind] ##have to normalize!
                        predicted_prob_straight = lda.predict_proba(features)[1]
                        if predicted_prob_straight > flip_threshold: # if frame is upside-down
                            #flip frame
                            M = cv2.getRotationMatrix2D((int(150/2),int(150/2)),180,1) 
                            frame_masked_straight = cv2.warpAffine(frame_masked_cropped,M,(crop_size,crop_size)) 
                            velocity_cur[frame_num-1,:] = -1 * velocity_cur[frame_num-1,0:2]
                            
                    else: #if not straightening, or already flipped use this frame as is
                        break
                
                
                # Display images
                if show_images:
                    if flip == 1:
                        frame = frame.copy()
                        cv2.putText(frame,'flip!',(10,130),0,1,255)
                    cv2.imshow('normal image',frame)
                    cv2.imshow('wavelet reconstruction',wavelet_recon)
                    
                    
                # ----------------------------------------------------
                # Stop video when finished and notify every 500 frames
                # ----------------------------------------------------
                if (frame_num)%5000==0:
                    print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')        
                if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
                    break
                if vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame:
                    break 
                
            else:
                print('broken...')
                
        # -------------
        # Save wavelets
        # -------------
        vid.release()
        if save_wavelets:
            save_file = file_location_data + upside_down_suffix + '_wavelet'
            coeff_save_file = file_location_data + upside_down_suffix + '_wavelet_slices'
            
            if os.path.isfile(save_file) and do_not_overwrite:
                raise Exception('File already exists')
            np.save(save_file,wavelet_array)
            np.save(coeff_save_file,coeff_slices)
    else:
        straighten_upside_down_frames = False #frames weren't straightened if wavelet analysis not performed
        
        
    if save_concatenate_data:
        if straighten_upside_down_frames:
            np.save(file_location_data + '_velocity_corrected.npy', velocity_cur) #save flipped velocity
        velocity = np.concatenate((velocity, velocity_cur),axis=0)
        out_of_bounds = np.append(out_of_bounds, out_of_bounds_cur)
        disruption = np.append(disruption, disruption_cur)
        frames = np.append(frames, frames_cur)
        coordinates = np.append(coordinates, coordinates_cur)

# ------------------------------------------
# save concatenated / flipped velocity, etc. 
# ------------------------------------------
if save_concatenate_data and not upside_down:
    if os.path.isfile(file_location_concatenated_data + '_frames.npy') and do_not_overwrite:
        raise Exception('File already exists')
        
    np.save(file_location_concatenated_data + '_out_of_bounds.npy', out_of_bounds)
    np.save(file_location_concatenated_data + '_disruption.npy', disruption)
    np.save(file_location_concatenated_data + '_frames.npy', frames)
    np.save(file_location_concatenated_data + '_coordinates.npy', coordinates)
    if straighten_upside_down_frames:
        np.save(file_location_concatenated_data + '_velocity_corrected.npy', velocity)
    else:
        np.save(file_location_concatenated_data + '_velocity.npy', velocity)
     

    
       

    
    

