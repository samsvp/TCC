import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import sobel
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import rotate
from skimage.segmentation import watershed


# https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
def get_edges(img: np.ndarray, blur=False) -> np.ndarray:
    """
    Get edges from image using canny edge detector. 
    Set "blur" to True if the algorithm is finding too many edges
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)

    edges = canny(img_gray / 255)
    return edges


def get_markers(img_gray: np.ndarray,
        low_thresh = 30, high_thresh = 150) -> np.ndarray:

    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    markers = np.zeros_like(img_gray)
    markers[img_gray < low_thresh] = 1
    markers[img_gray > high_thresh] = 2
    return markers


def get_segmentation(img: np.ndarray, 
        low_thresh = 30, high_thresh = 150,
        return_labels=False) -> np.ndarray:
    """
    Segments objects from image
    """

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    markers = get_markers(img_gray, low_thresh, high_thresh)
    elevation_map = sobel(img_gray)

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    if return_labels:
        labeled_img, _ = ndi.label(segmentation)
        return segmentation, labeled_img
    else:
        return segmentation


def filter_label(_img: np.ndarray, _img_labels: np.ndarray,
        label: int) -> np.ndarray:
    """
    Retrieves the object which was assigned the given label
    """
    img = _img.copy()
    img_labels = _img_labels.copy()

    img_labels[img_labels != label] = 0
    img_labels[img_labels == label] = 255

    rows_mask = (img_labels != 0).any(axis=1)
    img = img[rows_mask, :]
    img_labels = img_labels[rows_mask, :]

    cols_mask = (img_labels != 0).any(axis=0)
    img = img[:, cols_mask]
    img_labels = img_labels[:, cols_mask]
    
    # return the mask as alpha channel
    return np.append(img, img_labels[:,:,None], axis=-1).astype(np.uint8)
