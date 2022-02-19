# %%
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

data_dir = "torch_training"

dataset = ImageFolder(data_dir, 
    transform = transforms.Compose([
        transforms.ToTensor()
]))


labels_map = {
    0: "No Oil",
    1: "Oil"
}


batch_size = 128
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size 

train_data,val_data = random_split(dataset,[train_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")


#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3)
        self.fc1 = nn.Linear(1120, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss(weight=torch.tensor([5.0]))
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels.float().reshape(-1,1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:
            print(f'[{epoch + 1}, {i + 1}] loss: {loss}')
            running_loss = 0.0

print('Finished Training')

# %%

TILE_X = 120
TILE_Y = 160
IMG_SHAPE = (480, 640)
TILE_NUMBER = (IMG_SHAPE[0]//TILE_X, IMG_SHAPE[1]//TILE_Y)

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


def get_bounding_boxes(img: np.ndarray, net) -> np.ndarray:
    """
    Draws the predicted bounding boxes in the image
    """
    img = img.copy()
    for y in range(TILE_NUMBER[0]):
        for x in range(TILE_NUMBER[1]):
            tile = get_tile(img, x, y)
            tile = torch.tensor(np.transpose(tile, (2, 1, 0))[None, ...])/1.0
            out = net(tile)
            
            if out[0] <= 0.25:
                continue
            
            img = bounding_box(img, 
                y * TILE_Y, x * TILE_X, TILE_Y, TILE_X)
    return img

"""
Video bounding boxes
"""
video_path = "video/90graus-10m-17-11_13-44_2.MOV"

i = 0
cap = cv2.VideoCapture(video_path)
# Read until video is completed
with torch.no_grad():
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        i += 1
        if i < 200: continue
        
        if ret:
            frame = cv2.resize(frame, IMG_SHAPE[::-1],
                interpolation=cv2.INTER_CUBIC)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.medianBlur(frame, 5)
            frame = get_bounding_boxes(
                frame, net)
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()
# %%
