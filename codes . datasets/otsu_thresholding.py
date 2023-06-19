import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# import image
img = imread("blood_temp.jpg")
# plt.imshow(img)
print(f"Original Image: {img}")

# convert the image to grayscale
gray_img = rgb2gray(img)
# plt.imshow(gray_img, cmap="gray")
print(f"Grayscaled Image: {gray_img}")

# generate image histogram
hist = np.histogram(gray_img.flatten())
fig, ax = plt.subplots(ncols=2, figsize=(10,5))

ax[0].hist(gray_img.flatten())
ax[1].imshow(gray_img, cmap="gray")

# OTSU THRESHOLDING 
def threshold_otsu_impl(image, nbins=0.2):
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


threshold_c = threshold_otsu_impl(gray_img)
bin_img = gray_img > threshold_c

threshold_i = threshold_otsu(gray_img)
bin_img = gray_img > threshold_i

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].set_title("Original image")
ax[0].imshow(gray_img, cmap="gray")

ax[1].set_title("Binary image")
ax[1].imshow(bin_img, cmap="gray")
plt.imsave("bin_img.jpg", bin_img, cmap=plt.cm.gray)




# EDGE DETECTION
# import skimage.filters
  
# # applying gaussian blur
# gblur_bin_img = skimage.filters.gaussian(bin_img)
# # applying laplacian edge detection on the blurred image
# laplacian_bin_img = skimage.filters.laplace(gblur_bin_img)
# plt.imsave("gblur.jpg", gblur_bin_img, cmap=plt.cm.gray)

# fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
# ax[0].set_title("Gaussian image")
# ax[0].imshow(gblur_bin_img)

# ax[1].set_title("Laplacian image")
# ax[1].imshow(laplacian_bin_img, cmap="gray")