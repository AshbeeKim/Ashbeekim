'''import library'''
import os
import sys
import shutil
from glob import glob as gb

import itertools
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ResNet50
from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

'''
tensorflow version is 2.6.0
codes are not stabled
'''

class MultiClass():
  '''
  this class for tensorflow, I already load images and saved .npy file, just fit to my code
  텐서플로우에서 구동되도록 작성한 함수. jpg에서 바로 불러오는 부분은 npy 파일로 대체함
  '''
  def __init__(self, **params):
    self.dataframe = pd.read_csv(params['annotation_file'])
    self.npy_img = np.load(params['npy_path'])
    self.cols = params['cols']
    self.nums = params['nums']
    try:
      self.transform = params['transform']
    except:
      self.transform = None
    try:
      self.target_transform = params['target_transform']
    except:
      self.target_transform = None

  def __len__(self):
    return len(self.npy_img)

  def get_labels(self):
    max_num = self.nums['max']
    self.dataframe = (self.dataframe.sort_values(by=self.cols['label'])).reset_index(drop=True)
    self.img_labels = self.dataframe.loc[:, self.cols['label']].copy()
    self.img_classes = self.dataframe.loc[:, self.cols['target']].copy()
    self.cls_dict = {unq:num for num, unq in enumerate(np.unique(self.img_classes))}
    self.classes = np.array(self.img_classes.map(self.cls_dict)[:max_num])
    return self.img_labels, self.cls_dict, self.classes

  def get_images(self):
    pixels = self.nums['pixel']
    max_num = self.nums['max']
    if self.npy_img.shape[1]==pixels:
      images = self.npy_img[:max_num,...]
    else: # 
      images = tf.image.resize(self.npy_img[:max_num,...], [pixels, pixels], method="bicubic")
    self.images = tf.cast(images, dtype=tf.float32) / 255.0
    return self.images

  def preprocessing(self):
    self.images = self.get_images()
    self.cls_dict = self.get_labels()[1]
    self.classes = self.get_labels()[2]
    self.multi_cls = tf.cast(to_categorical(self.classes, num_classes=len(self.cls_dict)), dtype=tf.int32)
    self.dataset = self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.multi_cls))
    return self.cls_dict, self.dataset

class MCLS_Model(tf.keras):  #ResNet50, ~101, ~152, ~50V2, ~101V2, ~152V2
  def __init__(self, dataset, cls_dict, **params): # parameters
    self.dataset = dataset  # tensor형태
    self.input_shape = tuple(self.dataset.element_spec[0].shape)  # Dataset's 0번째 데이터 :  이미지
    self.cls_dict = cls_dict
    self.num_classes = len(self.cls_dict.values())
    self.params = params
    self.batch_size = params["batch_size"]
    self.epochs = params["epochs"]
    # self.images, self.labels = next(iter(self.dataset))
  
  def MCLS_Build(self): # uncompiled model
    # activation
    repeat_act = self.params["repeat"]  #relu
    last_act = self.params["last"]  #softmax
    # base model
    self.resnet = ResNet50(include_top=False, 
                           weights = "imagenet",
                           input_shape=self.input_shape)
    # new model on top
    self.inputs = Input(shape=self.input_shape)
    

  def MCLS_Compile(self): # compiled model
    loss = self.params["loss"] # "sparse_categorical_crossentropy"
    optimizer = self.params["optimizer"]# "SGD" #
    if type(self.params["metrics"])==list:
      metrics = self.params["metrics"]
    else:
      metrics = [self.params["metrics"]]

    self.model = self.MCLS_Build()  # call uncompiled model
    self.model.compile()
    return self.model

  def MCLS_History(self): # validation_split=0.1로 하면 KFOLD로 들어감. epochs을 높이면 불균형데이터 무관, shuffle도 가능, 
    self.history = self.MCLS_Compile().fit()
