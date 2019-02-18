# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import os
import openpyxl



print(sys.executable)
#Setting fixed threshold criteria
USE_THRESH = True
#fixed threshold value
THRESH = 10.0
THRESH_H = 300.0

#Video path of the source file
videopath = os.path.dirname(__file__)
#Directory to store the processed frames
dir = 'Keyframes'
#smoothing window size
len_window = 1


def rel_change(a, b):
    if (a - b == 0 ) :
        return 0
    if not a == 0 and b == 0:
        return 1
    x = float(np.absolute(b - a)) / b
    return x


def keyframe_extraction(videofile, video_num, video_name, videopath, tdir, len_window, info_file):

    keyframe_info_wb = openpyxl.load_workbook(info_file)
    keyframe_info_ws = keyframe_info_wb['keyframe']
    row_num = keyframe_info_ws.max_row

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
                name = "frame_" + str(i) + ".jpg"
                keyframe_info_ws.append([i, "frame_" + str(i), video_num, video_name])
                print(sdir + name)
                cv2.imwrite('dataset' + '/' + sdir + name, frame)

        prev_frame = curr_frame

        i = i + 1
        ret, frame = cap.read()



    cap.release()

    keyframe_info_wb.save('keyframe_info.xlsx')


print("Video :" + videopath)
print("Frame Directory: " + dir)


video_info_wb = openpyxl.load_workbook('video_info.xlsx')
video_info_ws = video_info_wb['info']
sample_num = video_info_ws.max_row - 1

keyframe_info_wb = openpyxl.Workbook('video_info.xlsx')
keyframe_info_ws = keyframe_info_wb.create_sheet('keyframe')
keyframe_info_ws.append(['frame number', 'frame file', 'video number', 'video name'])

keyframe_info_wb.save('keyframe_info.xlsx')


for i in range(sample_num):
    video_num = str(video_info_ws['A'+str(i+2)].value)
    video_name = str(video_info_ws['B'+str(i+2)].value)
    videofile = str(video_info_ws['C'+str(i+2)].value)

    print(video_num +", " + video_name+', '+videofile)
    keyframe_extraction(videofile, video_num, video_name, videopath, dir, len_window, 'keyframe_info.xlsx')
