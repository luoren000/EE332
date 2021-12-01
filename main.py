import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt



def crop_img(path):

    img = cv.imread(path)
    cv.imshow("Image", img)
    points = []
    def click_event(event, x, y, flags, params):
            nonlocal points
            if event == cv.EVENT_LBUTTONDOWN:
                points = [(x,y)]
            if event==cv.EVENT_LBUTTONUP:
                points.append((x,y))
            cv.rectangle(img, points[0], points[1], (255, 0, 0), 1)

    cv.setMouseCallback('Image', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # crop image
    if len(points) == 2:
        min_x = min(points[0][1], points[1][1])
        max_x = max(points[0][1], points[1][1])
        min_y = min(points[0][0], points[1][0])
        max_y = max(points[0][0], points[1][0])
        crop = img[min_x+1:max_x, min_y+1:max_y]
        cv.imwrite('headshots/9.jpg', crop)
        cv.imshow("cropped image", crop)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return points

def Frames_to_Video(name,path, fps):
    img = cv.imread(path+'1200.jpg')
    #cv.imshow('img', img)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(name + '.mov', fourcc, fps, (img.shape[1], img.shape[0]))

    for i in range(1201,3000):
        file_name = path + f'{i}.jpg'
        img = cv.imread(file_name)
        video.write(img)

    #cv.destroyAllWindows()
    video.release()

def Video_to_Frames(path):

    vidcap = cv.VideoCapture(path)
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      cv.imwrite("image/%d.jpg" % count, image)     # save frame as JPEG file
      if cv.waitKey(10) == 27:                     # exit if Escape is hit
          break
      count += 1

#points = crop_img("image/12124.jpg")

dic = {"red":[0,0,255],
       "blue":[255,0,0]}

TrDict = {'csrt': cv.legacy.TrackerCSRT_create,
         'kcf' : cv.legacy.TrackerKCF_create,
         'boosting' : cv.legacy.TrackerBoosting_create,
         'mil': cv.legacy.TrackerMIL_create,
         'tld': cv.legacy.TrackerTLD_create,
         'medianflow': cv.legacy.TrackerMedianFlow_create,
         'mosse':cv.legacy.TrackerMOSSE_create}

mode = 'medianflow'

'''img_rgb = cv.imread('image/1200.jpg')
img = img_rgb.copy()
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

trackers = cv.legacy.MultiTracker_create()

directory = r'headshots/'
for filename in os.listdir(directory):
    color, number = filename.split("_")
    number = number.split(".")[0]
    #print(color, number)

    template = cv.imread('headshots/{}'.format(filename))
    height, width = template.shape[0], template.shape[1]
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF )
    #cv.imshow("res", res)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    tracker_i = TrDict[mode]()
    bbi = top_left + (width,) + (height,)
    #bbi = cv.selectROI('Frame', img)
    #print(top_left)
    #print(bottom_right)
    #print(bbi)
    trackers.add(tracker_i, img, bbi)
    cv.rectangle(img_rgb, top_left, bottom_right, dic[color], 2)
    cv.putText(img_rgb, f'{number}', top_left, cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv.LINE_AA)

isExist = os.path.exists('results/'+mode)
if not isExist:
    os.makedirs('results/'+mode)
cv.imwrite('results/{}/1200.jpg'.format(mode), img_rgb)

for i in range(1201, 3000):
    file_name = 'image/{}.jpg'.format(i)
    train_img = cv.imread(file_name)
    (success, boxes) = trackers.update(train_img)
    for box,filename in zip(boxes,os.listdir(directory)):
        color, number = filename.split("_")
        number = number.split(".")[0]
        (x, y, w, h) = [int(a) for a in box]
        cv.rectangle(train_img, (x, y), (x + w, y + h), dic[color], 2)
        cv.putText(train_img, f'{number}', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imwrite('results/{}/{}.jpg'.format(mode,i), train_img)'''


#cv.imshow("Matched image", img_rgb)
#cv.waitKey()
#cv.destroyAllWindows()

Frames_to_Video(mode,'results/{}/'.format(mode),60)