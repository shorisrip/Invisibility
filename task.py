import numpy as np
import cv2

#
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('out.avi', fourcc, 20.0, size)

background = 0  # capturing background
for i in range(30):
    ret, background = cap.read()  # capturing image

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([95,10,18])
    upper_red = np.array([135,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # lower_red = np.array([70, 15, 10])
    # upper_red = np.array([110, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # mask1 = mask1 + mask2  # OR
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                             iterations=2)

    # mask2 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8),
    #                          iterations=1)

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(final_output)

    cv2.imshow('CV2 Window', final_output)
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()