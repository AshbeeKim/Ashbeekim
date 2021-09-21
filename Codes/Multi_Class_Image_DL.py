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
