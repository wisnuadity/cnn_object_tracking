import sys
sys.path.append("..")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# exit()
# from utils import DataGenerator, read_annotation_lines
from utils import read_annotation_lines
import os
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LeakyReLU
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
'''
['0.7326283987915403', '92.06948640483384', '34.0', '114.3936436635395']
['1.6389728096676777', '89.95468277945619', '34.0', '116.50083789652567']
['0.7326283987915403', '89.04833836858006', '34.0', '116.54680159963677']
'''
im_w = 64
im_h = 384
def DataGenerators(data_lines):
    img_list = []
    bbox_list = []
    for lines in data_lines:
        lines_str1 = lines.split(" ")
        lines_str2 = lines_str1[1].split(",")
        img = cv2.imread("img/all/"+lines_str1[0])/255
        print("img/all/"+lines_str1[0])
        h1,w1,c1 = img.shape
        print("top:", (int(float(lines_str2[0])), int(float(lines_str2[1]))))
        print("bot:", (int(float(lines_str2[2])), int(float(lines_str2[3]))))
        center_coordinates1 = (int(float(lines_str2[0])), int(float(lines_str2[1])))
        center_coordinates2 = (int(float(lines_str2[2])), int(float(lines_str2[3])))
        # center_coordinates2 = (20, 100)
        radius = 1
        color1 = (255, 0, 0)
        color2 = (255, 255, 0)
        thickness = 8
        img_show = img.copy()
        img_show = cv2.circle(img_show, center_coordinates1, radius, color1, thickness)
        img_show = cv2.circle(img_show, center_coordinates2, radius, color2, thickness)
        img_show = cv2.resize(img_show, (150, 700))
        # cv2.imshow("in img", img_show)
        # cv2.waitKey(0)
        img = cv2.resize(img,(im_w,im_h))
        # img = cv2.resize(img,(64,64))
        if w1<im_w:
            ws = w1/im_w
        elif w1>=im_w:
            ws = im_w/w1
        if h1<im_h:
            hs = h1/im_h
        elif h1>=im_h:
            hs = im_h/h1
        # print(lines_str2[0:4])
        # continue
        # print(img.shape)

        img_list.append(img)
        bbox_list.append([float(lines_str2[0])*ws,float(lines_str2[1])*hs,float(lines_str2[2])*ws,float(lines_str2[3])*hs])

        # print(float(lines_str2[0])*ws,float(lines_str2[1])*hs,float(lines_str2[2])*ws,float(lines_str2[3])*hs)
        # bbox_list.append(lines_str2[0:4])
    # print(len(img_list),img_list)
    dataImg = np.asarray(img_list)
    dataBbox = np.asarray(bbox_list)
    return dataImg, dataBbox
    # return img_list, bbox_list

train_lines, val_lines = read_annotation_lines('img/ano/anotation.txt', test_size=0.5)
img_train, bbox_train = DataGenerators(train_lines)
img_val, bbox_val = DataGenerators(val_lines)
# exit()
print(img_train.shape)
# exit()

FOLDER_PATH = '../all/0'
class_name_path = '../class_names/wl.txt'
# data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
# data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)

data_gen_train = DataGenerators(train_lines)
data_gen_val = DataGenerators(val_lines)
'''
model = Yolov4(weight_path=None,
               class_name_path=class_name_path)


'''
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(im_h,im_w,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(128, kernel_size=1, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(256, kernel_size=1, activation='relu'))
model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(512, kernel_size=1, activation='relu'))
# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

model.add(Flatten())
model.add(Dense(512))
# model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation=LeakyReLU(alpha=0.1)))

optimizer = Adam(lr=0.002)
# model2.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
model.summary()

model.fit(img_train, bbox_train,
# model.fit(data_gen_train,
          initial_epoch=0,
          epochs=100,
          batch_size=36,
          validation_data=(img_val, bbox_val),
          # val_data_gen=data_gen_val,
          shuffle=True,
          callbacks=[])
model.save("detect4.keras")

new_model = load_model('detect4.keras')
# img_test, bbox_test = DataGenerators(val_lines)
print(img_val.shape,img_val.shape[0])
for i in range(int(img_val.shape[0])):
    pred_input = img_val[i]
    pred_input = pred_input.reshape((1,im_h,im_w,3))
    img = (img_val[i])*255
    img = (img).astype("uint8")
    res = new_model.predict(pred_input)
    center_coordinates1 = (int(res[0][0]), int(res[0][1]))
    center_coordinates2 = (int(res[0][2]), int(res[0][3]))
    radius = 1
    color1 = (255, 0, 0)
    color2 = (255, 255, 0)
    thickness = 1
    img = cv2.circle(img, center_coordinates1, radius, color1, thickness)
    img = cv2.circle(img, center_coordinates2, radius, color2, thickness)
    img = cv2.resize(img,(300,500))
    cv2.imshow("res",img)
    cv2.waitKey(0)
    # print(res)