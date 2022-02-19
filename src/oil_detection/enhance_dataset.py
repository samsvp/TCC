# %%
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift


#%%
dir = "training"
files = [f"{dir}/{f}" for f in os.listdir(dir)]

imgs = [f for f in files if "img" in f]
masks = [f for f in files if "mask" in f]

imgs.sort()
masks.sort()

l = len(imgs)

def crop(img, x, y, w, h):
    return img[y:y+h,x:x+w,:]


# %%
# rotate
for i in range(100):

    index = random.randint(0, l-1)
    angle = random.uniform(0.0, 360.)

    img = cv2.imread(imgs[index])
    mask = cv2.imread(masks[index])

    img = rotate(img, angle)
    mask = rotate(mask, angle)

    cv2.imwrite(f"synth_{i}_img.png", img)
    cv2.imwrite(f"synth_{i}_mask.png", mask)

    if i%50 == 0:
        print(f"saved synth_{i}_img.png")


# shift
for i in range(100, 150):
    index = random.randint(0, l-1)
    angle = random.uniform(0.0, 360.)

    img = cv2.imread(imgs[index])
    mask = cv2.imread(masks[index])

    shift_x = random.uniform(50., 150.)
    shift_y = random.uniform(50., 150.)

    img = shift(img, (shift_y, shift_x, 0))
    mask = shift(mask, (shift_y, shift_x, 0))

    cv2.imwrite(f"synth_{i}_img.png", img)
    cv2.imwrite(f"synth_{i}_mask.png", mask)

    if i%50 == 0:
        print(f"saved synth_{i}_img.png")


# crop
for i in range(150, 250):
    index = random.randint(0, l-1)
    angle = random.uniform(0.0, 360.)

    img = cv2.imread(imgs[index])
    mask = cv2.imread(masks[index])

    x = random.randint(0, int(0.5*img.shape[0]))
    y = random.randint(0, int(0.5*img.shape[1]))
    w = img.shape[0] - x
    h = img.shape[1] - y

    img_cropped = crop(img, x, y, w, h)
    mask_cropped = crop(mask, x, y, w, h)
    img = cv2.resize(img_cropped, img.shape[:-1])
    mask = cv2.resize(mask_cropped, mask.shape[:-1])

    cv2.imwrite(f"synth_{i}_img.png", img)
    cv2.imwrite(f"synth_{i}_mask.png", mask)

    if i%50 == 0:
        print(f"saved synth_{i}_img.png")


for i in range(250,750):
    changed = False

    index = random.randint(0, l-1)
    angle = random.uniform(0.0, 360.)

    img = cv2.imread(imgs[index])
    mask = cv2.imread(masks[index])

    # rotation
    if random.uniform(0.0,1.0) > 0.1:
        img = rotate(img, angle)
        mask = rotate(mask, angle)
        changed = True


    # shift
    if random.uniform(0.0,1.0) > 0.2:
        shift_x = random.uniform(50., 150.)
        shift_y = random.uniform(50., 150.)

        img = shift(img, (shift_y, shift_x, 0))
        mask = shift(mask, (shift_y, shift_x, 0))

        changed = True


    # crop
    if random.uniform(0.0,1.0) > 0.3:
        x = random.randint(0, int(0.5*img.shape[0]))
        y = random.randint(0, int(0.5*img.shape[1]))
        w = img.shape[0] - x
        h = img.shape[1] - y

        img_cropped = crop(img, x, y, w, h)
        mask_cropped = crop(mask, x, y, w, h)
        img = cv2.resize(img_cropped, img.shape[:-1])
        mask = cv2.resize(mask_cropped, mask.shape[:-1])

        changed = True

    if changed:
        cv2.imwrite(f"synth_{i}_img.png", img)
        cv2.imwrite(f"synth_{i}_mask.png", mask)

        if i%50 == 0:
            print(f"saved synth_{i}_img.png")


# for _ in range(200):
#     index = random.randint(0, l-1)
#%%


# %%
pixels = []
for m in masks:
    img = cv2.imread(m)
    img[img<127] = 0
    img[img>=127] = 255
    pixels.append(np.unique(img, return_counts=1)[-1])
# %%
p_black, p_white = zip(*[(p[0], p[1]) if len(p) == 2 else (p[0], 0)
    for p in pixels])

print(sum(p_black) / sum(p_white)) # ~180
# %%
