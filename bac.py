import sys
import cv2
import numpy as np
cv2.CascadeClassifier
def nothing(x):
    pass

# Create a window to display the camera feed
cv2.namedWindow('Camera Output',cv2.WINDOW_NORMAL)
cv2.namedWindow('Hand',cv2.WINDOW_NORMAL)
#cv2.namedWindow('HandTrain',cv2.WINDOW_NORMAL)   

# cascade xml file for detecting palm. Haar classifier
palm_cascade = cv2.CascadeClassifier('palm3.xml')
videoFrame = cv2.VideoCapture(0)
_, prevhandImg = videoFrame.read()
prevcnt = np.array([], dtype=np.int32)

# previous values of cropped variable
x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0
keyPressed = -1

while(keyPressed):      

    min_YCrCb = np.array([0,130,103], np.uint8)
    max_YCrCb = np.array([255,182,130], np.uint8)

     # Grab video frame, Decode it and return next video frame
    readSuccess, sourceImage = videoFrame.read()

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours by area. Largest area first.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get largest contour and compare with largest contour from previous frame.
    # set previous contour to this one after comparison.
    cnt = contours[0]
    #ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]
    cnt=contours[0]
    # once we get contour, extract it without background into a new window called handTImg
    stencil = np.zeros(sourceImage.shape).astype(sourceImage.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [cnt], color)
    handTImg = cv2.bitwise_and(sourceImage, stencil)


    # crop coordinates for hand.
    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

    # place a rectange around the hand.
    cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

    # if the crop area has changed drastically form previous frame, update it.
    if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
        x_crop_prev = x_crop
        y_crop_prev = y_crop
        h_crop_prev = h_crop
        w_crop_prev = w_crop

    # create crop image
    handImg = sourceImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

    # Training image with black background
    handTImg = handTImg[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                     max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]



    # haar cascade classifier to detect palm and gestures
    gray = cv2.cvtColor(handImg, cv2.COLOR_BGR2HSV)
    palm = palm_cascade.detectMultiScale(gray)
    for (x, y, w, h) in palm:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_color = sourceImage[y:y + h, x:x + w]

    # to show convex hull in the image
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # counting defects in convex hull. To find center of palm. Center is average of defect points.
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if count_defects == 0:
            center_of_palm = far
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            if count_defects < 5:
                # cv2.circle(sourceImage, far, 5, [0, 0, 255], -1)
                center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
        cv2.line(sourceImage, start, end, [0, 255, 0], 2)
    # cv2.circle(sourceImage, avr, 10, [255, 255, 255], -1)

    # drawing the largest contour
    cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)
    

    # Display the source image and cropped image
    camera = cv2.VideoCapture(0)
    camera.set(10,200)
    cv2.imshow('Camera Output', sourceImage)
    cv2.imshow('Hand', handImg)
    cv2.imshow('HandTrain', handTImg)
    keyPressed = cv2.waitKey(30) 

cv2.destroyAllWindows() 
videoFrame.release()
