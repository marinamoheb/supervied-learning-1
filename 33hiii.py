#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.datasets import mnist

import numpy as np
from PIL import Image, ImageOps
import os

def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


DIR_NAME = "JPEGImages"
if os.path.exists(DIR_NAME) == False:
    os.mkdir(DIR_NAME)

# Save Images
i = 0
for li in [x_train, x_test]:
    print("[---------------------------------------------------------------]")
    for x in li:
        filename = "{0}/{1:05d}.jpg".format(DIR_NAME,i)
        print(filename)
        save_image(filename, x)
        i += 1


# In[16]:


def Load ():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


# In[17]:


def crop(data,h,w):
   traindata = []
   for image in range(len(data)):
       im = PIL.Image.fromarray(np.uint8(data[image]))
       (image_width, image_height) = im.size
       rows = np.int(image_height / h)
       cols = np.int(image_width / w)
       feature_vector = []
       for i in range(rows):
           for j in range(cols):
               box = (j * w, i * h, (j + 1) * w, (i + 1) * h)
               block = im.crop(box)
               X_centroid, Y_centroid = centroid(block)
               feature_vector.append(X_centroid)
               feature_vector.append(Y_centroid)
       feature_vector_train = np.array(feature_vector)
       traindata.append(feature_vector_train)
   traindata = np.vstack(traindata)
   return traindata


# In[45]:


#another way to divid images
import os
import sys
from PIL import Image

savedir = r"C:JPEGImages"
filename = r"C:JPEGImages/00000.jpg"
img = Image.open(filename)
width, height = img.size
start_pos = start_x, start_y = (0, 0)
cropped_image_size = w, h = (14,14)

frame_num = 1
for col_i in range(0, width, w):
    for row_i in range(0, height, h):
        crop = img.crop((col_i, row_i, col_i + w, row_i + h))
        save_to= os.path.join(savedir, "testing_{:02}.png")
        crop.save(save_to.format(frame_num))
        frame_num += 1


# In[12]:


def centroid(block):
    block = np.asarray(block)
    sum_x, sum_y, fun = 0, 0, 0
    for X in range(block.shape[0]):
        for Y in range(block.shape[1]):
            sum_x = sum_x + (X * block[X][Y])
            sum_y = sum_y + (Y * block[X][Y])
            fun =fun + block[X][Y]
    X_centroid = sum_x / fun if fun > 0 else 0
    Y_centroid = sum_y / fun if fun > 0 else 0
    return X_centroid, Y_centroid


# In[14]:


import PIL
import numpy as np
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
x_train, y_train, x_test, y_test = Load()
x_train = crop(x_train,5,5)
x_test = crop(x_test,5,5)
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train, y_train)
y_pred  = knn.score(x_test, y_test )
print( y_pred )


# In[15]:


knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train, y_train)
y_pred  = knn.score(x_test, y_test )
print( y_pred )


# In[ ]:




