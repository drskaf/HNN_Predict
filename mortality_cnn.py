import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from datetime import datetime
from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy, losses_utils
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2
import tempfile
from keras import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, concatenate, add, Input, Conv2D, MaxPool2D, BatchNormalization, \
    AveragePooling2D, GlobalAveragePooling2D, Activation, ZeroPadding2D

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 64
NUM_EPOCHS = 500
N_CLASSES = 1

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])

# Info file
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/Labels.csv')

# Loading images and labels
(df) = utils.load_image_png(args["directory"], label_file, INPUT_DIM)

# class_weight = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# print(class_weight)
class_weight = {0: 0.53042993,
                1: 8.71559633}

# Splitting data
(df_train, df_valid) = train_test_split(df, train_size=0.7, stratify=df[args["target"]])
X_train = np.array([x for x in df_train['images']]) / 255.0
X_valid = np.array([z for z in df_valid['images']]) / 255.0
y_train = np.array(df_train.pop(args["target"]))
tlist = y_train.tolist()
print(tlist.count(1))
y_valid = np.array(df_valid.pop(args["target"]))
vlist = y_valid.tolist()
print(vlist.count(1))
print(y_train[:10])
print(y_valid[:10])

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

def build_model(output_bias=None, reg=0.0002):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential()
    inputShape = (HEIGHT, WIDTH, 1)
    chanDim = -1

    # if we are using "channels first", update the input shape
    # and channels dimension
    if K.image_data_format() == "channels_first":
        inputShape = (1, HEIGHT, WIDTH)
        chanDim = 1

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(20, kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))

    # sigmoid classifier
    model.add(Dense(N_CLASSES, activation='sigmoid', bias_initializer=output_bias))

    # Tune the learning rate for the optimizer
    model.compile(optimizer=Opt,
                  loss=Loss,
                  metrics=METRICS)

    return model


weigth_path = "{}_my_model.best.hdf5".format("image_mortality_LeNet")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=30,
    mode='max',
    restore_best_weights=True)

image_model = build_model()
print(image_model.predict(X_train[:10]))

initial_bias = np.log([0.053])
model = build_model(output_bias=initial_bias)
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

image_model = build_model()
image_model.load_weights(initial_weights)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
# print("[INFO] Training the model ...")

history = image_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=BATCH_SIZE,
                          epochs=NUM_EPOCHS,
                          callbacks=[early_stopping, checkpoint, tensorboard_callback], class_weight=class_weight)

# summarize history for loss
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train precision', 'validation precision', 'train loss', 'validation loss'], loc='upper left')
plt.show()

# Saving model data
model_json = image_model.to_json()
with open("image_mortality_LeNet.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
