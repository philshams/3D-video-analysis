# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:05:42 2018

@author: SWC
"""
import numpy as np; import nptdms as td; import cv2

 
    
#read tdms file of short video
file_path = ''

tdms_file = td.TdmsFile(file_path)
group = tdms_file.groups()
channel = tdms_file.group_channels(group[0])
channel_object = tdms_file.object(group[0],'data')
data = channel_object.data

video = np.reshape(data,(len(data)/1024/1048,1024,1280))
cv2.imshow('frame',video[0,:,:])





   
   