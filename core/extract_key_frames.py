# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import openpyxl
from scipy.signal import argrelextrema



print(sys.executable)
#Setting fixed threshold criteria
USE_THRESH = True
#fixed threshold value
THRESH = 10.0
THRESH_H = 300.0


def rel_change(a, b):
    if (a - b == 0 ) :
        return 0
    if not a == 0 and b == 0:
        return 1
    x = float(np.absolute(b - a)) / b
    return x




def keyframe_extraction(videofile, video_num, video_name, videopath, tdir):


    cap = cv2.VideoCapture(str(videopath)+"/"+str(videofile))

    sdir = tdir+"/"+str(video_num)

    curr_frame = None
    prev_frame = None

    count = 0 
    prev_count = 0

    ret, frame = cap.read()
    i = 1

    while(ret):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            prev_count = count
            count = np.sum(diff)

            relative_change = rel_change(np.float(prev_count), np.float(count))

            if(relative_change > THRESH_H):
                continue

            if (  relative_change >= THRESH ):
                print(relative_change)
                #print("prev_frame:"+str(frames[i-1].value)+"  curr_frame:"+str(frames[i].value))
                name = '_' + str(i) + ".jpg"
                print(sdir + name)
                cv2.imwrite(sdir + name, frame)
                i = i + 1


        prev_frame = curr_frame
        ret, frame = cap.read()



    cap.release()

def kf_main() :
    videopath = './dataset/Videos'
    dir = './dataset/Keyframes'
    video_info_wb = openpyxl.load_workbook('./dataset/video_info.xlsx')
    video_info_ws = video_info_wb['info']

    sample_num = video_info_ws.max_row - 1
    for i in range(sample_num):
	    video_num = str(video_info_ws['A'+str(i+2)].value)
	    video_name = str(video_info_ws['B'+str(i+2)].value)
	    videofile = str(video_info_ws['C'+str(i+2)].value)

	    print(video_num +", " + video_name+', '+videofile)
	    keyframe_extraction(videofile, video_num, video_name, videopath, dir)