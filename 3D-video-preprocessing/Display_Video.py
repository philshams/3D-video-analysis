'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                                          Display a saved video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import cv2


# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
file_name = 'ay117b_test1.avi' #
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '14.03.2018_zina\\' #
mouse_session = 'test_video\\'  #

file_loc = file_loc + date + mouse_session

time = np.load(file_loc + 'ay117b_test1_timestamps.npy')
print(file_loc)

display_frame_rate = 20
start_frame = 0
end_frame = 10000


# -----------
# Play video
# -----------
vid = cv2.VideoCapture(file_loc + file_name)  
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

while True:
    ret, frame = vid.read() # get the frame

    if ret: 
        cv2.imshow('movie',frame)
        
        frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
        print(frame_num)

        if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
            break
        
        if (frame_num-start_frame)%500==0:
            print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
            
        if frame_num >= min(end_frame,vid.get(cv2.CAP_PROP_FRAME_COUNT)):
            break 
    else:
        print('Problem with movie playback')
        cv2.waitKey(1000)
        
vid.release()
# Display number of last frame
print('Stopped at frame ' + str(frame_num)) #show the frame number stopped at



 