import cv2 as cv
import os

# name = input()

# ctr = 0
# cv.namedWindow("test")

## code to create data directory
# DIR = r'./Resources/Faces/train'
# if name not in  os.listdir(DIR):
#     os.mkdir(os.path.join(DIR,name))


cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("test", frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cap.release()

cv.destroyAllWindows()
 