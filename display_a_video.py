# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:54:54 2018

@author: SWC
"""
import cv2

file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'

file_loc = file_loc + date + mouse_session
#file_loc = ''

frame_rate = 20
display_frame_rate = 20

#cv2.destroyAllWindows()
#%%

vid = cv2.VideoCapture(file_loc + 'test_snip_data.avi')  

start_frame = 1
end_frame = 10000
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)


frame_no = start_frame
ret, frame = vid.read() # get the frame
print(ret)

while True:
    ret, frame = vid.read() # get the frame

    if ret: 
        
        cv2.imshow('normal',frame)
        
        frame_no +=1

        if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
            break
        if vid.get(cv2.CAP_PROP_POS_FRAMES) >= min(end_frame,vid.get(cv2.CAP_PROP_FRAME_COUNT)):
            break 
        
vid.release()



 