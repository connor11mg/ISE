import numpy as np
import cv2
##THIS PART FOR MASKING AND COLOR SEPARATION
#
# #original image
# ori = cv2.imread("shapesColors2.png")
#
# #converting original image to hsv format
# hsv = cv2.cvtColor(ori, cv2.COLOR_BGR2HSV)
#
# #HSV color array using numpy array
# lower_yellow = np.array([20,65,0])
# upper_yellow = np.array([32,255,255])
#
# lower_blue = np.array([60, 100, 0])
# upper_blue = np.array([100, 255, 255])
#
# lower_green = np.array([35, 100, 0])
# upper_green = np.array([60, 255, 255])
#
# #mask with lower and upper HSV values on HSV converted image
# mask = cv2.inRange(hsv, lower_green, upper_green)
#
# #showing results by applying mask on original image
# result = cv2.bitwise_and(ori, ori, mask=mask)
#
# cv2.imshow("Original", ori)
# cv2.imshow("HSV_Original", hsv)
# cv2.imshow("Mask", mask)
# cv2.imshow("yellow", result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

##FUNCTION FOR TRACKBAR FOR UPPER AND LOWER BOUNDARIES
def nothing(x):
    pass

# Load image
image = cv2.imread('shapesColors2.png')

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()