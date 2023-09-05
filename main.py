import cv2
import os
from cvzone import HandDetector

# Variables
width, height = 200, 100
folderPath = "slides"
imageNumber = 0

# camera Setup

# pylint: disable=no-member
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of presentation images.
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# set our video capture
hs, ws = int(120 * 1), int(213 * 1)

# hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:

    # importing images
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[imageNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # detecting Hands.
    hands, img = detector.findHands(img)

    # adding webCam image on the slide.
    imageSmall = cv2.resize(img, (ws, hs))  # resize the image

    # getint the size of the slides
    h, w, _ = imageSmall.shape
    imgCurrent[0:h, w - ws:w] = imageSmall

    cv2.imshow("Image", img)
    cv2.imshow("Image Current", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
