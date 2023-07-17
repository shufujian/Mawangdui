# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %% --------------------------------------- Data Prep -----------------------------------------------------------------

# Read data
DIR = './data/BAGAN_data/'

train = [f for f in os.listdir(DIR)]
train.pop(3)
print(train)
# train_sorted = sorted(train, key=lambda x: int(x[5:-4]))
trainsort = []
for f in train:
    for j in os.listdir(DIR+'/'+f):
        if j == '.ipynb_checkpoints':
            continue
        trainsort.append(DIR+f+'/'+j)

imgs = []
texts = []
resize_to = 64
print(len(trainsort))
for f in trainsort:
    # if f[-3:] == 'png':
    imgs.append(cv2.resize(cv2.imread(f), (resize_to, resize_to)))
    texts.append(f[18:19])
imgs = np.array(imgs)
texts = np.array(texts)
le = LabelEncoder()
le.fit(train)
labels = le.transform(texts)
print(labels)

# Splitting
SEED = 42
x_train, x_val, y_train, y_val = train_test_split(imgs, labels,
                                                  random_state=SEED,
                                                  test_size=0.2,
                                                  stratify=labels)
print(type(x_train))
print(type(imgs))

# %% --------------------------------------- Save as .npy --------------------------------------------------------------
# Save
np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val); np.save("y_val.npy", y_val)
# np.save("x_train.npy", imgs); np.save("y_train.npy", labels)