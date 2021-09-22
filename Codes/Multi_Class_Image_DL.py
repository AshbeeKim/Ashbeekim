'''
tensorflow version : 2.6.0
tensorflow-gpu version : 2.6.0
'''
############################################# default
import os
import sys
import time
import shutil
from os.path import join as jn
from glob import glob as gb

import itertools
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

############################################# image
import skimage.io as io
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import cv2 as cv

############################################## tensorflow
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # used for converting labels to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

############################################# pytorch
import torch
from torch import permute
from torchvision import transforms

############################################# Class & Def
'''
image
'''
class DuoDirsImages():	# faster than OpenCV
  def __init__(self, path, dir1=None, dir2=None, img_type='jpg'):
    if (dir1!=None)&(dir2==None):
      self.imgs = sorted(gb(f'{path}/{dir1}/*.{img_type}'))
    elif (dir1==None)&(dir2!=None):
      self.imgs = sorted(gb(f'{path}/{dir2}/*.{img_type}'))
    elif (dir1!=None)&(dir2!=None):
      self.path = gb(f'{path}/{dir1}/*.{img_type}')
      for paths in gb(f'{path}/{dir2}/*.{img_type}'):
        self.path.append(paths)
      self.imgs = sorted(self.path)
      self.labels = [lbl.split('/')[-1].split('.')[0] for lbl in self.imgs]
  
  def __len__(self):
    return len(self.imgs)

  def __ScikitImages__(self, start=0, end=1000):
    self.images = []
    for num in range(start, end):
      img_path = self.imgs[num]
      self.images.append(io.imread(img_path))
    return np.array(self.images)
'''How to load
KG_PATH = "/content/drive/MyDrive/Kaggle/SkinCancerMnist"
dir_list = [jn(KG_PATH, "HAM10000_images_part_1"), jn(KG_PATH, "HAM10000_images_part_2")]
duodirs = DuoDirsImages(KG_PATH, dir1=dir_list[0], dir2=dir_list[1])
images_10015 = duodirs.__ScikitImages__(0,10015) 
'''

class As_CV():
  def __init__(self, raw_data):
    self.data = raw_data
    self.cv_image = self.__ToOpenCV__()

  def __ToOpenCV__(self):
    self.cv_image = img_as_ubyte(self.data)
    return self.cv_image
  
  def __ResizeCV__(self, pixels):
    for num, image in enumerate(self.cv_image): #(512,512,3)
      self.resized_image = cv2.resize(image, (pixels, pixels), cv2.INTER_AREA)
      if num==0:
        self.cv_resized = np.array([self.resized_image])
      else:
        self.cv_resized = np.append(self.cv_resized, [self.resized_image], axis=0)
    return self.cv_resized
'''How to resize and save
images_10015 = As_CV(images_10015).__ResizeCV__(128)
HAM = jn(KG_PATH, "HAMNIST")
np.save(f"{HAM}/HAM_128_resized", images_10015)
'''



'''
tensorflow
'''
class MultiClass():
  '''
  jpg에서 바로 불러오는 부분은 npy 파일로 대체함
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

'''How to Start
cols = {'label':'image_id', 'target':'dx'}
nums = {'pixel':224,'max':1000}

init_params = {"annotation_file" : gb(KG_PATH+"/*metadata.csv")[0],
               "npy_path" : gb(HAM+"/*128*.npy")[0],
               "cols":cols, "nums":nums}

MCLS = MultiClass(**init_params)
HAM_cls_dict, HAM_dataset = MCLS.preprocessing()
'''

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

# 아직 공식문서를 통한 코드 이해가 부족한 관계로 우선은 해당 내용까지만 작성
'''How to train

'''



'''
pytorch
'''
def NHWC_to_NCHW(images):
  tensored = torch.FloatTensor(images)  # torch.Size([10015,256,256,3])
  permuted = tensored.permute(0,3,1,2)
  return permuted
# tensorflow class 작성 전에 구동한 코드라 확실하지 않음
'''How to use
permuted_HAM = NHWC_to_NCHW(resized_HAM)
# tensor로 바꿈
'''
