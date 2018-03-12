'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform Wavelet Transform on 3D mouse video                             --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
import pywt
import cv2
import os




#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
save_vid_name = 'analyze_7_3'


file_loc = 'C:\Drive\Video Analysis\data\\'
date = '14.02.2018_zina\\' #
mouse_session = 'twomouse\\'  #

save_vid_name = 'analyze'

# load video ...
file_loc = file_loc + date + mouse_session + save_vid_name

# ---------------------------
# Select analysis parameters
# ---------------------------
frame_rate = 1000
stop_frame = np.inf
show_images = True
save_data = False
do_not_overwrite = True

level = 5 # how many different spatial scales to use
discard_scale = 4 # 4 discards 4/5; 6 keeps all




#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Do wavelet transform                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ----------------------
# Set up video playback
# ----------------------
vid = cv2.VideoCapture(file_loc + '_data.avi')
num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
stop_frame = int(np.min([stop_frame, num_frames]))
wavelet_array = np.zeros((39,39,stop_frame)).astype(np.float16)
   

# ---------------------------------------------
# for each frame, perform wavelet decomposition
# ---------------------------------------------
while True:
    ret, frame = vid.read() # get the frame

    if ret: 
        # ---------------------------------------------
        # grab and, optionally, display each frame
        # ---------------------------------------------
        frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
        frame = frame[:,:,0]
        frame = cv2.resize(frame,(150,150)) #resize...
        if show_images:
            cv2.imshow('normal image',frame)
        
        # -----------------------------
        # extract wavelet coefficients
        # -----------------------------
        coeffs_lowpass = [[],[],[],[],[],[]]
        coeffs = pywt.wavedec2(frame, wavelet='db1',level = level)
        for i in range(level+1):
            #discard coefficients at too coarse of a spaital scale, as set by discard_scale
            if i < discard_scale:
                coeffs_lowpass[i] = coeffs[i]
            else:
                coeffs_lowpass[i] = [None,None,None]
        wavelet_recon = pywt.waverec2(coeffs_lowpass, wavelet='db1').astype(np.uint8)
        if show_images:
            cv2.imshow('wavelet reconstruction',wavelet_recon)
        
        #place coefficients in an array, and take coeff_slices index for later reconstruction       
        coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs[0:discard_scale])
        wavelet_array[:,:,frame_num-1] = coeff_array
        
        
        # ----------------------------------------------------
        # Stop video when finished and notify every 500 frames
        # ----------------------------------------------------
        if (frame_num)%500==0:
            print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')        
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break
        if vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame:
            break 
        
    else:
        print('broken...')
        
# ----------
# Save data
# ----------
vid.release()
if save_data:
    save_file = file_loc + '_wavelet'
    coeff_save_file = file_loc + '_wavelet_slices'
    
    if os.path.isfile(save_file) and do_not_overwrite:
        raise Exception('File already exists')
    np.save(save_file,wavelet_array)
    np.save(coeff_save_file,coeff_slices)

