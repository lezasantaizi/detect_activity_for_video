# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os


def find_and_save_active_image(video_path):
    video_capture = cv2.VideoCapture(video_path)
    # video_capture = cv2.VideoCapture("/Users/chen/Downloads/新建文件夹/2017-05-26/02/29.mp4")
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    #为了速度快,做了scale
    scale = 0.5
    success, frame = video_capture.read()
    last_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    last_gray = cv2.resize(last_gray,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_CUBIC)

    #取出局部图片做检测
    last_gray = last_gray[int(200* scale):,int(400*scale):int(-200*scale)]

    detect_interval = int(fps)
    frame_idx = 0
    window_size = 3#保证连续3s的图片都是动作,防止错误检测
    window_arr = [0] * window_size
    flag = 0
    while success:
        cur_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_CUBIC)
        # cv2.imshow("a",cur_gray)
        frame_idx += 1

        #每秒做一次检测
        if frame_idx % detect_interval == 0:
            cur_gray = cv2.GaussianBlur(cur_gray,(5,5),0)
            last_gray = cv2.GaussianBlur(last_gray,(5,5),0)
            # last_gray = cv2.medianBlur(last_gray,9)
            cur_gray = cur_gray[int(scale*200):,int(scale*400):int(scale * -200)]

            diff = abs(cur_gray.astype("int16") - last_gray.astype("int16"))
            diff[diff < 3  ] = 0
            last_gray = cur_gray

            # print frame_idx / int(fps),":",np.sum(diff)
            window_arr[frame_idx%window_size] = np.sum(diff) > 3000
            if np.sum(window_arr) == window_size  :
                if flag == 0:
                    flag = 1
                    name_split = video_path.split("/")
                    cv2.imwrite("/Users/chen/PycharmProjects/helloworld/image/"+name_split[-3]+"_"+name_split[-2]+"_"+name_split[-1].split(".")[0]+"_"+str(int(frame_idx / detect_interval))+".jpg",frame)
            else:
                flag = 0
            # cv2.imshow("b",abs(cur_gray))
        success , frame = video_capture.read()

    cv2.destroyAllWindows()
    video_capture.release()


for parent,dirnames,filenames in os.walk("/Users/chen/Downloads/新建文件夹/2017-05-26"):
    for dir in dirnames:
        for sec_parent,sec_dirs,sec_filenames in os.walk(parent+os.sep + dir):
            for filename in sec_filenames:
                complete_file_name = sec_parent+ os.sep + filename
                find_and_save_active_image(complete_file_name)




