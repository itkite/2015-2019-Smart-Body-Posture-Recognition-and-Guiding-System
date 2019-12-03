import cv2
camera = cv2.VideoCapture(0)
camera.set(10,200)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    #padding = 10
    #cv2.rectangle(img,(x-padding,y-padding),(x+w+padding,y+h+padding),(255,0,0),2)

    #cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
     #          (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)
    

def get_images_to_compare():
    images_to_compare = []
    while True:
        comp_img = raw_input("jj.jpg")
        if len(comp_img) <= 1:
            # break if someone just hits enter
            break
        images_to_compare.append(comp_img)
    return images_to_compare

def main():
    #capture_img = "/Users/Me/home1.png"
    capture_img = input('mm.jpg')
    #img_to_compare = "/Users/Me/Documents/python programs/compare/img2.jpg"
    take_and_save_picture(capture_img)
    #### you have some odd var names here, basic gist, add a for loop
    for comp_image in get_images_to_compare():
        diff = compute_edges_diff(im1, im2)
        print ("Difference (percentage):"), diff
        if diff > 0.5:
            print (im1)
        else:
            print (im2)



