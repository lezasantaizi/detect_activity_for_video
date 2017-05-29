# -*- coding: utf-8 -*-

import cv2
import numpy as np

# video_capture = cv2.VideoCapture("/Users/chen/Downloads/新建文件夹/2017-05-26/02/29.mp4")
video_capture = cv2.VideoCapture(0)
fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))


feature_params = dict(maxCorners = 500,qualityLevel = 0.3,minDistance = 7,blockSize = 7)

lk_params = dict(winSize = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

color = np.random.randint(0,255,(100,3))


def draw_flow(img,flow, step = 16):
    h ,w = img.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    f = flow[y,x]
    # fx,fy = np.split(f,2,1)
    lines = np.vstack([x,y,x+f[:,0],y+f[:,1]]).T.reshape(-1,2,2)
    lines = np.int32(lines+0.5)
    vis = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #     cv2.polylines(frame,[np.int32(tr) for tr in tracks],False,(0,255,0))

    cv2.polylines(vis,lines,0,(0,255,0))
    # for (x1,y1),(x2,y2) in lines:
    #     cv2.circle(vis,(x1,y1),1,(0,255,0),-1)

    return vis


def draw_hsv(flow):

    h,w = flow.shape[:2]
    fx,fy = flow[:,:,0],flow[:,:,1]
    ang=  np. arctan2(fy,fx) + np.pi
    v = np.sqrt(fx*fx + fy * fy)
    hsv = np.zeros((h,w,3),np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4,255)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def warp_img(img,flow):
    h,w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res =cv2.remap(img,flow,None,cv2.INTER_LINEAR)
    return res


success, frame = video_capture.read()
last_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
last_gray = cv2.resize(last_gray,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_CUBIC)

# p0 = cv2.goodFeaturesToTrack(last_gray,mask= None,**feature_params)
# mask = np.zeros_like(frame)
tracks = []
track_len = 10
detect_interval = 5
frame_idx = 0
while success:
    cur_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.resize(cur_gray,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_CUBIC)

    # if frame_idx % detect_interval == 0:
        # if len(tracks) > 0:
    img0 ,img1 = last_gray,cur_gray
    # p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
    flow = cv2.calcOpticalFlowFarneback(img0,img1,0.5,3,15,3,5,1.2,0)
    last_gray = cur_gray



    frame_idx += 1

    cv2.imshow("a",draw_flow(cur_gray,flow,8))
    # cv2.imshow("a",draw_hsv(flow))
    # cv2.imshow("a",warp_img(cur_gray,flow))
    cv2.imshow("b",cur_gray)
    k = cv2.waitKey(1000/int(fps)) & 0xff
    if k == 27:
        break
    success , frame = video_capture.read()

cv2.destroyAllWindows()
video_capture.release()
