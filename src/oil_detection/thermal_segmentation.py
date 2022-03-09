#%%
import cv2
import numpy as np

import matplotlib.pyplot as plt
from skimage.filters import sobel


TILE_X = 512 // 2
TILE_Y = 640 // 2


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalizes the array so that vmin is 0 and vmax is size(dtype)
    """
    img = img.astype(float) - img.min()
    norm_img = img / img.max()
    return norm_img


def crop(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Returns a cropped image that matches its thermal counterpart
    """
    zoom_mask = cv2.resize(mask, (2326,1832), interpolation=cv2.INTER_AREA)
    
    offset_x = 0
    offset_y = 0

    height, width, channels = img.shape
    center_x, center_y = height // 2, width // 2

    cmin_x = center_x - zoom_mask.shape[0] // 2 + offset_x
    cmax_x = center_x + zoom_mask.shape[0] // 2 + offset_x
    cmin_y = center_y - zoom_mask.shape[1] // 2 + offset_y
    cmax_y = center_y + zoom_mask.shape[1] // 2 + offset_y

    frame = img[cmin_x:cmax_x, cmin_y:cmax_y]
    frame = cv2.resize(frame, mask.shape[::-1])
    return (frame, mask)


def get_tile(img: np.ndarray, x: int, y: int, 
        x_tile_size=1, y_tile_size=1) -> np.ndarray:
    """
    Returns the tile at position x and y on the img.
    Consecutive tiles can be returned with x_tile_size and y_tile_size
    Note: The first and last tiles in the y axis are ignored due to their size
    """
    tile_x1 = TILE_X * x
    tile_x2 = TILE_X * (x + x_tile_size)
    tile_y1 = TILE_Y * y
    tile_y2 = TILE_Y * (y + y_tile_size)
    return img[tile_x1:tile_x2, tile_y1:tile_y2]


def get_borders(_image: np.ndarray, thresh=0.0025,
        erode=True) -> np.ndarray:
    image = cv2.blur(_image,(15,15))
    norm = normalize(image)
    borders = cv2.blur(sobel(norm),(5,5))
    borders[borders >= thresh] = 1
    borders[borders < thresh] = 0
    if erode:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(borders, kernel)
    return borders


# %%
rets, images = cv2.imreadmulti(
    'video/20201210_180338_SEQ.TIFF', [],
    cv2.IMREAD_UNCHANGED)

#%%
for image in images:
    borders = get_borders(image)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(borders)
    plt.show()

# %%
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('thermal_out.avi', 
    fourcc, 24.0, (1280, 512))
mov = cv2.VideoCapture(
    'video/20201210_180338_VIS_H264.MOV')

for i, image in enumerate(images):
    borders = (255 * get_borders(image)).astype(np.uint8)
    
    # sync rgb video
    mov.set(cv2.CAP_PROP_POS_FRAMES, int(3.948 * i))
    ret, frame = mov.read()

    cframe, ctiff = crop(frame, borders)
    # put images side by side
    fframe = np.concatenate(
        (cframe, np.tile(ctiff[...,None], 3)), axis=1)
    out.write(fframe)

out.release()
mov.release()
# %%
mov = cv2.VideoCapture(
    'video/20201210_180338_VIS_H264.MOV')

for i, tiff in enumerate(images):
    borders = (255 * get_borders(image)).astype(np.uint8)
    
    # sync rgb video
    mov.set(cv2.CAP_PROP_POS_FRAMES, int(3.948 * i))
    ret, frame = mov.read()

    cframe, ctiff = crop(frame, borders)
    # put images side by side
    fframe = np.concatenate(
        (cframe, np.tile(ctiff[...,None], 3)), axis=1)

mov.release()
# %%
