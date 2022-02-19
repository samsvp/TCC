"""
Latest idea: use haralick to capture texture info
and train a ML algo to predict if there is or isnt oil
in a given region. After predictng, confirm it by running
edge detection on the proposed are and kmeans to isolate
the oil. This can be used to get a bigger dataset 
for the neural network
"""


# %%
import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import sobel
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import segmentation as seg



def norm_img(img: np.ndarray) -> np.ndarray:
    """
    Returns the image where the lower value is 0
    and the maximum is 255
    """
    img = (img - img.min())
    img = img / img.max() * 255
    return img.astype(np.uint8)

# %%
img_path = "training/34_img.png"
#img_path = "training/2_img.png"
mask_path = img_path.replace("img", "mask")

gray = True
blur = False
edge_params = (30, 150)


_img = cv2.imread(img_path)
_mmask = cv2.imread(mask_path)
if blur:
    _img = cv2.GaussianBlur(_img, (5,5), 0)

# sharpen
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#_img = cv2.filter2D(_img, -1, kernel)

img = _img
#img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
seg_img = cv2.cvtColor(_img, 
    cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
_seg, labeled_img = seg.get_segmentation(seg_img, *edge_params, True)

# show image and extracted labels
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(img)
axarr[1].imshow(labeled_img)
plt.tight_layout()
plt.show()

labeled_filter = seg.filter_label(img, labeled_img, 1)

# check if we filtered the images
plt.imshow(labeled_filter)
plt.tight_layout()
plt.show()
# %%
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
markers = seg.get_markers(norm_img(img_gray), 100, 150)
elevation_map = sobel(img_gray)
segmentation = watershed(elevation_map)
ndi.binary_fill_holes(elevation_map)
plt.imshow(elevation_map)
# %%
t = 5
melevation_map = norm_img(elevation_map)
melevation_map[melevation_map<t] = 0
melevation_map[melevation_map>=t] = 1
melevation_map.astype(bool)
plt.imshow(melevation_map)
# %%
mimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
vectorized = mimg.reshape((-1,3)).astype(np.float32)

attempts=50
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
     50, 1.0)
K = 3

ret,label,center=cv2.kmeans(vectorized,K,None,
    criteria,attempts,cv2.KMEANS_PP_CENTERS) 

center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((mimg.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.imshow(result_image)
plt.show()
# %%
_,thresh = cv2.threshold(img_gray, np.mean(img_gray), 
    255, cv2.THRESH_BINARY_INV)

edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

cnt = sorted(cv2.findContours(edges, 
    cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], 
    key=cv2.contourArea)[-1]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

dst = cv2.bitwise_and(img, img, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# %%
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(hsv[:,:,2], cmap="gray")

# %%
value = hsv[:,:,2].copy()
#value[value<150] = 0
#value[value>220] = 0
plt.imshow(value)
cv2.imshow('window',value)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
# %%
saturation = hsv[:,:,1].copy()
#saturation[saturation<20] = 0
# saturation[saturation>220] = 0
plt.imshow(saturation)

cv2.imshow('window',saturation)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

# %%
plt.imshow(img)
plt.show()
# %%
# this haralick shit seems to work
# training a classifier seems worth it
import mahotas
base_image = img_gray[350:450,350:450]
features = mahotas.features.haralick(base_image).mean(axis=0)
plt.imshow(base_image)
plt.show()
# %%
def plot_patch(patch: np.ndarray, patch_mask: np.ndarray) -> float:
    """
    Plots the patch and prints the haralick features
    difference
    """
    pfeatures = mahotas.features.haralick(patch).mean(axis=0)
    p = ((features - pfeatures)**2).sum()
    print(p)

    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(patch)
    axarr[1].imshow(patch_mask)
    plt.tight_layout()
    plt.show()
    return p
# %%
patch = img_gray[350:450, 450:550]
patch_mask = _mmask[350:450, 450:550]
plot_patch(patch, patch_mask)

patch = img_gray[0:100, 500:600]
patch_mask = _mmask[0:100, 500:600]
plot_patch(patch, patch_mask)

patch = img_gray[350:450, 180:280]
patch_mask = _mmask[350:450, 180:280]
plot_patch(patch, patch_mask)

patch = img_gray[100:200, 100:200]
patch_mask = _mmask[100:200, 100:200]
plot_patch(patch, patch_mask)

plt.imshow(img)
plt.show()
# %%
_mpatch = img[350:450, 180:280,:]
mpatch = cv2.cvtColor(_mpatch, cv2.COLOR_RGB2HSV)
f, axarr = plt.subplots(1,3) 
axarr[0].imshow(_mpatch)
axarr[1].imshow(canny(mpatch[:,:,2]))



attempts=50
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
     attempts, 1.0)
K = 2
mimg = mpatch[:,:,2].reshape(-1,1).astype(np.float32)

ret,label,center=cv2.kmeans(mimg,K,None,
    criteria,attempts,cv2.KMEANS_PP_CENTERS) 

center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((mpatch[:,:,2].shape))

axarr[2].imshow(result_image)
plt.tight_layout()
plt.show()

# %%
_mpatch = img[100:200, 100:200]
mpatch = cv2.cvtColor(_mpatch, cv2.COLOR_RGB2HSV)
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(_mpatch)
axarr[1].imshow(canny(mpatch[:,:,2]))
plt.tight_layout()
plt.show()
# %%
