from tkinter import image_names
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

# read image
image = cv2.imread("blood_img.jpg")

# PREPROCESSING for OT
# denoise image
dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
# BGR to RGB
image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

## gaussian blur
# blur = cv2.GaussianBlur(image, (5, 5), 0)

# HSV split
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)
cv2.imwrite("H.jpg",h)
cv2.imwrite("S.jpg",s)
cv2.imwrite("V.jpg",v)

# grayscale image
# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("./gray_img.jpg", gray_img)

image = v
# generate image histogram
hist = np.histogram(image.flatten())
fig, ax = plt.subplots(ncols=2, figsize=(10,5))
ax[0].hist(image.flatten())
ax[1].imshow(image, cmap="gray")
plt.show()

# OTSU THRESHOLDING
def threshold_otsu_impl(image, nbins=0.1):
    # validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        return False
    
    # validate multicolored
    if np.min(image) == np.max(image):
        return False
    
    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image)+nbins, np.max(image)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
            print("trace:", within_class_variance, color_threshold)
            
    return least_variance_threshold

threshold = threshold_otsu_impl(image)
bin_img = image < threshold

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].set_title("Original image")
ax[0].imshow(image, cmap="gray")

ax[1].set_title("Binary image")
ax[1].imshow(bin_img, cmap="gray")
plt.imsave("bin_img.jpg", bin_img, cmap=plt.cm.gray)



# MATHEMATICAL MORPHOLOGY 
# Dilation - add pixels to edges || Erosion - remove pixels from edges

# HOUGH CIRCLE TRANSFORMATION