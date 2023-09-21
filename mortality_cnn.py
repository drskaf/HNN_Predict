import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback, LearningRateScheduler
from datetime import datetime
from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy, losses_utils
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2
import tempfile
from array import array
from keras import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, concatenate, add, Input, Conv2D, MaxPool2D, BatchNormalization, \
    AveragePooling2D, GlobalAveragePooling2D, Activation, ZeroPadding2D
import cv2
from keras.models import load_model

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 1344
BATCH_SIZE = 32
NUM_EPOCHS = 500
N_CLASSES = 1
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

  ''' Fine tuning step - VGG19 '''

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/final_surv.csv')

# Loading images and labels
(df1) = utils.load_label_png('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/peak_LV_images', label_file, INPUT_DIM)
print(len(df1))
(df2) = utils.load_lge_data('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_img', label_file, INPUT_DIM)
print(len(df2))
df = df1.merge(df2, on='ID')
print(len(df))

class_weight = {0: 0.595,
                1: 3.3130}

# Splitting data
(df_train, df_valid) = train_test_split(df, train_size=0.7, stratify=df[args["target"]])
X_train = np.array([x for x in df_train['images']])
print(X_train.shape)
X_valid = np.array([x for x in df_valid['images']])
print(X_valid.shape)
y_train = np.array(df_train.pop(args["target"]))
tlist = y_train.tolist()
print(tlist.count(1))
y_valid = np.array(df_valid.pop(args["target"]))
vlist = y_valid.tolist()
print(vlist.count(1))
print(y_train[:10])
print(y_valid[:10])

# Data augmentation
aug = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

v_aug = ImageDataGenerator()

# Initialise the optimiser and model
print("[INFO] compiling model ...")
Opt = Adam(lr=0.001)
Loss = BinaryCrossentropy()
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

weigth_path = "{}_my_model.best.hdf5".format("image_mortality_AlexNet")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=50,
    mode='max',
    restore_best_weights=True)
callback = LearningRateScheduler(scheduler)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
# print("[INFO] Training the model ...")
image_model = tf_cnns.AlexNet(HEIGHT, WIDTH, DEPTH, N_CLASSES, reg=0.0002)
image_model.compile(loss= Loss, optimizer=Opt, metrics=METRICS)
history = image_model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=v_aug.flow(X_valid, y_valid),
                          epochs=NUM_EPOCHS,
                          callbacks=[early_stopping, checkpoint, tensorboard_callback], class_weight=class_weight)

# summarize history for loss
plt.plot(history.history['prc'])
plt.plot(history.history['val_prc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('prc_loss')
plt.xlabel('epoch')
plt.legend(['train f1 curve', 'validation f1 curve', 'train loss', 'validation loss'], loc='upper right')
plt.show()

# Saving model data
model_json = image_model.to_json()
with open("image_mortality_AlexNet.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
