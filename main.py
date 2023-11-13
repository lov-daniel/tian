#General Libraries
import numpy
import pyautogui
import time
import mss
from matplotlib import pyplot as plt

#Image Processing Libraries
from PIL import Image
import pytesseract
import cv2

dimensions = pyautogui.size()



# Obtained from MSS documention

def imageProcessing(img, isolation):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    if isolation == True:
        blur = cv2.GaussianBlur(img,(5,5),0)
        threshold, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return img

def processSyllable(img):
    return pytesseract.image_to_string(img)


with mss.mss() as sct:
    # Part of the screen to capture
    monitorBounds = {"top": 0, "left": 0, "width": dimensions[0] , "height": dimensions[1]}
    syllableBounds = {"top": round(dimensions[0] / 3.15), "left": round(dimensions[1] / 1.64), "width": 75, "height": 75}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        syllableCapture = numpy.array(sct.grab(syllableBounds))

        # Display the pictures
        cv2.imshow("Syllable Capture", imageProcessing(syllableCapture, True))
        print(f"fps: {1 / (time.time() - last_time)}")

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break