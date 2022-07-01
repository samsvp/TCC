# %%
import os
import cv2
import mahotas
import numpy as np
from skimage.feature import canny
from skimage.filters import sobel
from sklearn.base import ClassifierMixin
from collections import Counter

import matplotlib.pyplot as plt

from typing import List, Tuple


TILE_X = 60
TILE_Y = 80
IMG_SHAPE = (480, 640)
TILE_NUMBER = (IMG_SHAPE[0]//TILE_X, IMG_SHAPE[1]//TILE_Y)


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Returns the intersection over union of the masks"""
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)


def load_image_mask(img_path: str) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    loads image and corresponding mask
    """
    mask_path = img_path.replace("img", "mask")
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_gray.shape != IMG_SHAPE:
        img_gray = cv2.resize(img_gray, IMG_SHAPE[::-1])
    img_mask = cv2.imread(mask_path)
    return (img, img_gray, img_mask)


def print_tiles(img: np.ndarray) -> None:
    """
    Shows the tiles current on img separated by black lines
    """
    mask = np.ones((img.shape[0], img.shape[1]), 
        dtype=np.uint8) * 255
    mask[::TILE_Y, :] = 0
    mask[:, ::TILE_X] = 0

    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('tiles', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def has_oil(img_mask, thresh: int=10) -> int:
    """
    Returns wheter the image has oil or not
    based on its mask
    """
    return int(np.sum(img_mask) > thresh)


def get_low_level_feats(img: np.ndarray, 
        bins: int = 16) -> np.ndarray:
    """
    Returns image color saturation and value information
    """
    def count(array: np.ndarray, target: np.ndarray): 
        """
        Counts occurences in a range
        """
        for k, v in Counter(array // bins).items():
            target[k] = v

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    s = img_hsv[:,:,1].reshape(-1)
    v = img_hsv[:,:,2].reshape(-1)
    
    s_counts = np.zeros(256 // bins)
    v_counts = np.zeros(256 // bins)

    count(s, s_counts)
    count(v, v_counts)

    if np.max(s_counts) != 0:
        s_counts /= np.max(s_counts)
        v_counts /= np.max(v_counts)

    return np.append(s_counts, v_counts)


def get_high_level_feats(img: np.ndarray) -> np.ndarray:
    """
    Return the haralick features of the image
    (texture information)
    """
    borders = np.array(canny(img), dtype=np.uint8)
    borders = cv2.resize(borders, (16, 16)).reshape(-1)
    features = mahotas.features.haralick(img).mean(axis=0)
    return np.append(borders, features)
    


def get_batch(img_path: str, thresh=10) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Gets a training batch composed of features and
    ground truth
    """
    img, img_gray, img_mask = load_image_mask(img_path)

    _targets: List[int] = []
    _features: List[int] = []

    # extract features from image tile
    for y in range(TILE_NUMBER[0]):
        for x in range(TILE_NUMBER[1]):
            tile = get_tile(img, x, y)
            if 0 in tile.shape: continue

            tile_gray = get_tile(img_gray, x, y)
            tile_mask = get_tile(img_mask, x, y)
            # ignore empty tiles
            try:
                h_feats = get_high_level_feats(tile_gray)
                l_feats = get_low_level_feats(tile)
                _feature = np.append(l_feats, h_feats)
            except ValueError:
                continue
            _features.append(_feature)
            _target = has_oil(tile_mask, thresh=thresh)
            _targets.append(_target)
    
    features = np.array(_features)
    targets = np.array(_targets)

    return (features, targets)


def get_img_tiles(img_path: str, thresh=10) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Gets a training batch composed of features and
    ground truth
    """
    img, img_gray, img_mask = load_image_mask(img_path)

    _targets: List[int] = []
    _tiles: List[np.ndarray] = []

    # extract features from image tile
    for y in range(TILE_NUMBER[0]):
        for x in range(TILE_NUMBER[1]):
            tile = get_tile(img, x, y)
            if 0 in tile.shape: continue

            tile_mask = get_tile(img_mask, x, y)
            _target = has_oil(tile_mask, thresh=thresh)
            _targets.append(_target)
            _tiles.append(tile)
    
    targets = np.array(_targets)
    tiles = np.array(_tiles)

    return (tiles, targets)


def bounding_box(img: np.ndarray, x: int, y: int,
        w: int, h: int) -> np.ndarray:
    """
    Draws a bounding box on the given position
    """
    img = img.copy()
    color = (0, 0 , 255) # green
    thickness = 2 # 2px
    return cv2.rectangle(img, (x, y), (x + w, y + h), 
        color, thickness)


def get_prediction_mask(_img: np.ndarray, 
        clf: ClassifierMixin, thresh=0.25) -> np.ndarray:
    """
    Returns the mask using color segmentation with 
    the predictions of the given classifier
    """
    img = _img.copy()
    mmask = np.zeros(img.shape[:2])
    for y in range(TILE_NUMBER[0]):
        for x in range(TILE_NUMBER[1]):
            tile = get_tile(img, x, y)
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            h_feats = get_high_level_feats(tile_gray)
            l_feats = get_low_level_feats(tile)
            features = np.append(
                 l_feats, h_feats).reshape(1, -1)
            
            if clf.predict_proba(features)[:,1] < thresh:
                continue
            
            mask = apply_oil_mask(img, x, y)
            
            tile_x1 = TILE_X * x
            tile_x2 = TILE_X * (x + 1)
            tile_y1 = TILE_Y * y
            tile_y2 = TILE_Y * (y + 1)
            mmask[tile_x1:tile_x2, tile_y1:tile_y2] = mask

    return mmask


def get_bounding_boxes(_img: np.ndarray, 
        clf: ClassifierMixin, thresh=0.25) -> np.ndarray:
    """
    Draws the predicted bounding boxes in the image
    """
    img = _img.copy()
    area = 0
    for y in range(TILE_NUMBER[0]):
        for x in range(TILE_NUMBER[1]):
            tile = get_tile(img, x, y)
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            h_feats = get_high_level_feats(tile_gray)
            l_feats = get_low_level_feats(tile)
            features = np.append(
                 l_feats, h_feats).reshape(1, -1)
            
            
            if clf.predict_proba(features)[:,1] < thresh:
                continue
            
            mask = apply_oil_mask(img, x, y, True)
            oil_area = np.sum(mask)
            #if oil_area <= 30:
            #if (oil_area:= np.sum(mask)) <= 30:
            #    continue
            area += oil_area
    
    position = (10, 50)
    cv2.putText(
        img, f"Oil Area {area}px",
        position, cv2.FONT_HERSHEY_SIMPLEX, 1, #font size
        (209, 80, 0, 255), #font color
        2) #font stroke

    return img


def get_oil_mask(img: np.ndarray) -> np.ndarray:
    """
    Returns the positions where there is oil
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    l_limit = np.array([100, 40, 0])
    h_limit = np.array([150, 100, 255])

    mask = cv2.inRange(img_hsv, l_limit, h_limit)
    return mask


def apply_oil_mask(img: np.ndarray, x: int, y: int,
        draw_box=False) -> np.ndarray:
    """
    Applies the oil mask in the specified image region
    """
    tile_x1 = TILE_X * x
    tile_x2 = TILE_X * (x + 1)
    tile_y1 = TILE_Y * y
    tile_y2 = TILE_Y * (y + 1)
    tile = img[tile_x1:tile_x2, tile_y1:tile_y2]
    mask = get_oil_mask(tile)
    img[tile_x1:tile_x2, tile_y1:tile_y2] += \
        np.array([100, 0, 0], dtype=np.uint8) * mask[..., None]
    
    if draw_box:
        img[tile_x1:tile_x2, tile_y1:tile_y1+3] = [0, 0, 255]
        img[tile_x1:tile_x1+3, tile_y1:tile_y2] = [0, 0, 255]
        img[tile_x1:tile_x2, tile_y2-3:tile_y2] = [0, 0, 255]
        img[tile_x2-3:tile_x2, tile_y1:tile_y2] = [0, 0, 255]
    
    return mask


# %%
folder = "training"
img_paths = [f"{folder}/{f}" for f in os.listdir(folder) 
    if "img" in f]
training_len = int(0.8 * len(img_paths))


#%%
print("Creating training dataset")
_X, _Y = zip(*[get_batch(img_path, 50) 
    for img_path in img_paths[:training_len]])
X = np.array([x for _x in _X for x in _x])
y = np.array([y for _y in _Y for y in _y])


print("Getting test dataset")
_X, _Y = zip(*[get_batch(img_path) 
    for img_path in img_paths[training_len:]])

X_test = np.array([x for _x in _X for x in _x])
y_test = np.array([y for _y in _Y for y in _y])


# test a bunch of classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Perceptron",
    "AdaBoost",
    "QDA"
]

classifiers = [
    KNeighborsClassifier(5),
    DecisionTreeClassifier(class_weight="balanced",
        random_state=0),
    RandomForestClassifier(class_weight="balanced",
        n_estimators=100, random_state=0),
    MLPClassifier(alpha=1, max_iter=10000, random_state=0),
    AdaBoostClassifier(random_state=0),
    QuadraticDiscriminantAnalysis()
]

scores = []
# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(f"Training {name}")
    clf.fit(X, y)
    print(f"Validating {name}")
    score = clf.score(X_test, y_test)
    scores.append((name, score))

scores.sort(key=lambda x: x[-1], reverse=True)
print("\nFinal scores:")
for name, score in scores:
    print(f"\t{name}: {score}")


from sklearn.metrics import confusion_matrix

y_pred = classifiers[2].predict(X_test)
confusion_matrix(y_pred, y_test)
#%%
### IOU calculation
# Get masks
folder = "test"
test_path = [f"{folder}/{f}" for f in os.listdir(folder) 
    if "img" in f]

images, images_gray, images_mask = zip(
    *[load_image_mask(img_path) 
        for img_path in test_path]
)

ious_d = {}
for t in np.arange(0.1, 1.0, 0.1):
    pred_masks = [
        get_prediction_mask(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
            classifiers[2], thresh=t)
        for img in images
    ]

    ious = [iou(pred_masks[i].astype(np.uint8), 
                images_mask[i][...,0]) 
            for i in range(len(pred_masks))]

    ious_d[t] = ious
    print(t, "\n", ious, np.mean(ious), np.median(ious))
    print("\n")


#%%

pred_masks = [
    get_prediction_mask(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
        classifiers[2], thresh=0.5)
    for img in images
]

for i in range(len(pred_masks)): 
    p_sum = np.sum(pred_masks[i]) / 255
    m_sum = np.sum(images_mask[i][...,0]) / 255
    diff = np.abs(p_sum - m_sum)/m_sum
    print(i, diff)
    f, axarr = plt.subplots(1,3)
    f.set_size_inches(18.5, 10.5)
    axarr[0].imshow(pred_masks[i])
    axarr[1].imshow(images_mask[i])
    axarr[2].imshow(images[i])
    plt.show()


l = []
for i in range(len(pred_masks)): 
    p_sum = np.sum(pred_masks[i]) / 255
    m_sum = np.sum(images_mask[i][...,0]) / 255
    diff = np.abs(p_sum - m_sum)/m_sum
    print(i, diff)
    l.append(diff)
print(l.mean(), l.std())


#%%
# Apply oil masks
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

pred_masks = [
    get_bounding_boxes(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
        classifiers[2], thresh=0.5)
    for img in images
]

for i in range(len(images)):
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.1, hspace=0.25)
    pl.figure().set_size_inches(12.5, 10.5)
    
    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    ax.imshow(images[i])
    ax.set_title("Original image")
    
    ax = pl.subplot(gs[0, 1]) # row 0, col 1
    ax.imshow(images[i] + np.array(
        images_mask[i] / 255 * 200, dtype=np.uint8))
    ax.set_title("Ground truth mask on original image")
    
    ax = pl.subplot(gs[1, :]) # row 1, span all columns
    ax.imshow(pred_masks[i])
    ax.set_title("Predicted truth mask on original image")
    


#%%
# raw image test
classifiers_img = [
    KNeighborsClassifier(5),
    DecisionTreeClassifier(class_weight="balanced",
        random_state=0),
    RandomForestClassifier(class_weight="balanced",
        n_estimators=100, random_state=0),
    MLPClassifier(alpha=1, max_iter=10000, random_state=0),
    AdaBoostClassifier(random_state=0),
    QuadraticDiscriminantAnalysis()
]

scores = []

print("Creating raw image training dataset")
_X, _Y = zip(*[get_img_tiles(img_path, 50) 
    for img_path in img_paths[:training_len]])
X_img = np.array([x.reshape(-1) for _x in _X for x in _x])
y_img = np.array([y.reshape(-1) for _y in _Y for y in _y])


print("Getting test dataset")
_X, _Y = zip(*[get_img_tiles(img_path) 
    for img_path in img_paths[training_len:]])

X_img_test = np.array([x.reshape(-1) for _x in _X for x in _x])
y_img_test = np.array([y.reshape(-1) for _y in _Y for y in _y])

# iterate over classifiers
for name, clf in zip(names, classifiers_img):
    print(f"Training {name}")
    clf.fit(X_img, y_img.reshape(-1))
    print(f"Validating {name}")
    score = clf.score(X_img_test, y_img_test.reshape(-1))
    scores.append((name, score))

scores.sort(key=lambda x: x[-1], reverse=True)
print("\nFinal scores:")
for name, score in scores:
    print(f"\t{name}: {score}")

# %%
#img_path = "training/34_img.png"
img_path = "training/50_img.png"
img, img_gray, img_mask = load_image_mask(img_path)
print(img_gray.shape)

plt.imshow(img_gray, cmap="gray")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for name, clf in zip(names, classifiers):
    plt.imshow(get_bounding_boxes(img, clf))
    
    # for tcc
    if name == "Random Forest":
        plt.savefig("random_forest50.png")
        
    plt.title(name)
    plt.show()

# %%
"""
Video bounding boxes
"""
video_path = "video/90graus-10m-17-11_13-44_2.MOV"

i = 0
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 24.0, IMG_SHAPE[::-1])
show_frame = False


while(cap.isOpened()):

    ret, mframe = cap.read()
    i += 1
    # if i < 100: continue
    
    if ret:
        mframe = cv2.resize(mframe, IMG_SHAPE[::-1],
            interpolation=cv2.INTER_CUBIC)
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.medianBlur(mframe, 5)
        frame = get_bounding_boxes(
            frame, classifiers[2], thresh=0.45)
        if show_frame:
            cv2.imshow('Frame', frame)
        
        # save the frame
        out.write(frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        if i % 100 == 0:
            print(i)
    else: 
        break


cap.release()
out.release()
cv2.destroyAllWindows()


# %%
img_path = "training/34_img.png"
img, img_gray, img_mask = load_image_mask(img_path)

img_hsv = cv2.cvtColor(mframe, cv2.COLOR_RGB2HSV)
s = img_hsv[:,:,1]
step = 15
for i in range(0, 256, step):
    s[(i <= s) & (s <= (i + step))] = i
plt.imshow(s)
plt.show()

v = img_hsv[:,:,2]
step = 10
for i in range(0, 256, step):
    v[(i <= v) & (v <= (i + step))] = i
plt.imshow(v)
plt.show()

# %%
plt.imshow(img)
# %%
tile = get_tile(frame, 3, 3)
#tile = get_tile(frame, 2, 2)

#tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
l_limit = np.array([100, 40, 180])
h_limit = np.array([150, 80, 255])

mask = cv2.inRange(img_hsv, l_limit, h_limit)
res = cv2.bitwise_and(tile, tile, mask=mask)
plt.imshow(tile)
plt.show()
plt.imshow(res)
plt.show()

# %%
def save_batch(img_path: str, i: int) -> \
        None:
    """
    Gets a training batch composed of features and
    ground truth
    """
    img, img_gray, img_mask = load_image_mask(img_path)

    # extract features from image tile
    for y in range(TILE_NUMBER[0]):
        for x in range(TILE_NUMBER[1]):
            tile = get_tile(img, x, y)
            if 0 in tile.shape: continue

            tile_mask = get_tile(img_mask, x, y)
            
            cv2.imwrite(f"torch_training/{i}_{x}{y}_img.png", tile)
            cv2.imwrite(f"torch_training/{i}_{x}{y}_mask.png", tile_mask)
    

for i, img_path in enumerate(img_paths):
    save_batch(img_path, i)
# %%
