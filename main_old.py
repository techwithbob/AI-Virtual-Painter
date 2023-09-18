import cv2
import numpy as np
import time
import os
import handtrack as htm

#######################
brushThickness = 25
eraserThickness = 100
########################

folderPath = "Template"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(imPath)
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
# Create a mask of logo
img2gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
x_header, y_header = 240, 0

drawColor = 0

cap = cv2.VideoCapture(0)
dim = (1080,720)
# cap.set(3, 1280)
# cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65,maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((dim[1], dim[0], 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.resize(img, dim)
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        print(x1)
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # 4. If Selection Mode – Two finger are up
        if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
            print("Selection Mode")
        # # Checking for the click
            if y1 < 125:
                if 280 < x1 < 340:
                    header = overlayList[0]
                    drawColor = 0 
                elif 400 < x1 < 440:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 520 < x1 < 560:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 640 < x1 < 680:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 750 < x1 < 800:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                    
        # 5. If Drawing Mode – Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            if drawColor == 0:
                cv2.circle(img, (x1, y1), 15, (255,255,255), cv2.FILLED)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

        xp, yp = x1, y1

        # Clear Canvas when all fingers are up
        if all (x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1080, 3), np.uint8)

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img,imgInv)
            img = cv2.bitwise_or(img,imgCanvas)

    # # Setting the header image
    img[y_header:200+y_header, x_header:600+x_header] = header
    # roi = img[y_header:200+y_header, x_header:600+x_header]
    # # Set an index of where the mask is
    # roi[np.where(mask)] = 0
    # roi += header
    
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    
cv2.destroyAllWindows()