import cv2
import numpy as np

def empty(a):
    print("")

cap = cv2.VideoCapture(0)
cv2.namedWindow("TrackBars")

cv2.createTrackbar("Hue min","TrackBars",68,179,empty)
cv2.createTrackbar("Hue max","TrackBars",110,170,empty)
cv2.createTrackbar("Sat min","TrackBars",55,255,empty)
cv2.createTrackbar("Sat max","TrackBars",255,255,empty)
cv2.createTrackbar("Val min","TrackBars",55,255,empty)
cv2.createTrackbar("Val max","TrackBars",255,255,empty)

while True:
    cv2.waitKey(1000)
    ret, init_frame = cap.read()
    if ret:
        break

while True:
    ret, frame = cap.read()
    imgHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat max","TrackBars")
    v_min = cv2.getTrackbarPos("Val min","TrackBars")
    v_max = cv2.getTrackbarPos("Val max","TrackBars")

    kernel = np.ones((3,3),np.uint8)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    mask = cv2.inRange(imgHSV,lower,upper)
    mask = cv2.medianBlur(mask,3)
    mask_inv = 255 - mask
    mask = cv2.dilate(mask,kernel,5)

    b = frame[:,:,0]
    g = frame[:,:,1]
    r = frame[:,:,2]
    b = cv2.bitwise_and(mask_inv,b)
    g = cv2.bitwise_and(mask_inv,g)
    r = cv2.bitwise_and(mask_inv,r)
    area = cv2.merge((b,g,r))

    b = init_frame[:, :, 0]
    g = init_frame[:, :, 1]
    r = init_frame[:, :, 2]
    b = cv2.bitwise_and(b,mask)
    g = cv2.bitwise_and(g,mask)
    r = cv2.bitwise_and(r,mask)
    blanket_area = cv2.merge((b, g, r))

    harry_cloak = cv2.bitwise_or(area,blanket_area)

    cv2.imshow("Original",frame)
    cv2.imshow("Harry Cloak",harry_cloak)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
