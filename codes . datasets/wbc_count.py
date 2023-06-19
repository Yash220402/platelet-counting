import numpy as np
import imutils
import cv2

image = cv2.imread('BloodImage_00000.jpg')

dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
# the meaning of parameters given
# p1 = 10: size of pixels to compute weights of the image
# p2 = 10: to compute the weighted average
# p3 = 7: filter strength for luminescence
# p4 = 15: filter strength for color component

rgb_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
cv2.imwrite("RGB_image.jpg",rgb_image)

new_image = cv2.medianBlur(rgb_image, 5)
cv2.imwrite("median_blur.jpg",new_image)

hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)
cv2.imwrite("H.jpg",h)
cv2.imwrite("S.jpg",s)
cv2.imwrite("V.jpg",v)

ret,th1 = cv2.threshold(h,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('Binary_image.jpg',th1)

kernel = np.ones((5,5), dtype = "uint8")/9
bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
erosion = cv2.erode(bilateral, kernel, iterations = 6)
cv2.imwrite('mask_erosion.jpg', erosion)

# find contours in the thresholded image
cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} unique contours found".format(len(cnts)))

# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the contour
    ((x, y), _) = cv2.minEnclosingCircle(c)
    cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imwrite('Contour_Image.jpg',image)