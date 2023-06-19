import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import gaussian

# IMAGE PREPROCESSING
image = imread("45.bmp")
# gaussian blur
gaus = gaussian(image)
# rgb to lab
labimg = rgb2lab(image, illuminant='A', observer='2')
# lab to gray
gray = rgb2gray(labimg)

from skimage.filters import threshold_minimum

thresh_min = threshold_minimum(gray)
binary_min = image > thresh_min

io.imshow(binary_min)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].set_title("Original image")
ax[0].imshow(image)
ax[1].set_title("Lab image")
ax[1].imshow(labimg)

# io.imshow(labimg)
# plt.show()

# IMAGE SEGMENTATION
# Otsu Thresholding
def threshold_otsu_impl(image, nbins=0.6):
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


threshold_c = threshold_otsu_impl(gray)
bin_img = gray > threshold_c

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].set_title("Original image")
ax[0].imshow(gray, cmap="gray")

ax[1].set_title("Binary image")
ax[1].imshow(bin_img, cmap="gray")
plt.show()
# cv2.waitKey(0)