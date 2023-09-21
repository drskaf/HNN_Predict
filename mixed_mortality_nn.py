import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import os
import argparse
import utils
import locale
import matplotlib.image as mpimg
from skimage.transform import resize
import tf_cnns
from keras.models import Sequential
from keras. layers import Input, Dense, Flatten, concatenate, Conv2D, Activation, MaxPool2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical, plot_model
from datetime import datetime
import visualkeras
from ann_visualizer.visualize import ann_viz
import graphviz
import pydot
from sklearn.preprocessing import StandardScaler
import tempfile
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import sample

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 224
DEPTH = 25
BATCH_SIZE = 32
NUM_EPOCHS = 500
STEP_PER_EPOCH = 50
N_CLASSES = 1

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/final_surv.csv')
patient_df['Gender'] = patient_df['patient_GenderCode_x'].astype('category')
patient_df['Gender'] = patient_df['Gender'].cat.codes

# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)_x','Essential_hypertension_x', 'Gender_x', 'Heart_failure_(disorder)_x', 'Smoking_history_x',
'Dyslipidaemia_x', 'Myocardial_infarction_(disorder)_x', 'Diabetes_mellitus_(disorder)_x', 'Cerebrovascular_accident_(disorder)_x']
numerical_col_list= ['Age_on_20.08.2021_x_x', 'LVEF_(%)_x']

# Defining networks
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    if regress:
	    model.add(Dense(1, activation="linear"))
    # return our model
    return model

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
  # Input:
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path

  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPool2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer

def GoogLeNet(INPUT_SHAPE, OUTPUT):
    # input layer
    input_layer = Input(INPUT_SHAPE)

    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # convolutional layer: filters = 64, strides = 1
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)

    # convolutional layer: filters = 192, kernel_size = (3,3)
    X = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(X)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # 1st Inception block
    X = Inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)

    # 2nd Inception block
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # 3rd Inception block
    X = Inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)

    # 4th Inception block
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                        f3_conv5=128, f4=128)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # 8th Inception block
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)

    # 9th Inception block
    X = Inception_block(X, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)

    # Global Average pooling layer
    X = GlobalAveragePooling2D(name='GAPL')(X)

    # Dropoutlayer
    X = Dropout(0.4)(X)

    # output layer
    X = Dense(OUTPUT, activation='relu')(X)

    # model
    model = Model(input_layer, X, name='GoogLeNet')

    return model

# Loading data
(df1) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_images', patient_df, INPUT_DIM)
print(len(df1))
(df2) = utils.load_lge_images('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_images', patient_df, INPUT_DIM)
print(len(df2))
df = df1.merge(df2, on='ID')
print(len(df))
perf_imgs = np.array([x for x in df['Perf']])
lge_imgs = np.array([x for x in df['LGE']])
Imgs = []
for p, l in zip(perf_imgs, lge_imgs):
    i = np.block([p,l])
    Imgs.append(i)
df['images'] = Imgs

class_weight = {0: 0.5896,
                1: 3.2892}

def process_attributes(df, train, valid):
    continuous = numerical_col_list
    categorical = categorical_col_list
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    valContinuous = cs.transform(valid[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    trainCategorical = catBinarizer.transform(train[categorical])
    valCategorical = catBinarizer.transform(valid[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    valX = np.hstack([valCategorical, valContinuous])

    return (trainX, valX)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
print("[INFO] processing data...")
(df_train, df_valid) = train_test_split(df, train_size=0.8, stratify=df[args["target"]])
trainy = np.array(df_train[args["target"]])
tlist = trainy.tolist()
print(tlist.count(1))
validy = np.array(df_valid[args["target"]])
vlist = validy.tolist()
print(vlist.count(1))
X_train = np.array([x for x in df_train['LGE']])
print(X_train.shape)
X_valid = np.array([x for x in df_valid['LGE']])
print(X_valid.shape)
print(trainy[:10])
print(validy[:10])

train_gen = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.1, zoom_range
                         =0.1, horizontal_flip=True, fill_mode="nearest")
trainImages = train_gen.flow(X_train, batch_size=2000)
trainImagesX = trainImages.next()

valid_gen = ImageDataGenerator()
validImages = valid_gen.flow(X_valid, batch_size=1000)
validImagesX = validImages.next()
(trainAttrX, validAttrX) = process_attributes(df, df_train, df_valid)

# create the MLP and CNN models
mlp = create_mlp(trainAttrX.shape[1], regress=False)
cnn = tf_cnns.GoogLeNet((HEIGHT, WIDTH, DEPTH), OUTPUT=4)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(N_CLASSES, activation="sigmoid")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (outcome
# prediction)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using binary categorical cross-entropy given that
# we have binary classes of either the prediction is positive or negative
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
opt = Adam(lr=1e-3)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=METRICS)
#print(model.predict([trainAttrX, trainImagesX][:10]))
weigth_path = "{}_my_model.best.hdf5".format("HNN_GoogleNet")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)
early_stopping = EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=50,
    mode='max',
    restore_best_weights=True)
callback = LearningRateScheduler(scheduler)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# train the model
print("[INFO] training model...")
history = model.fit(x=[trainAttrX, trainImagesX], y=trainy, validation_data=([validAttrX, validImagesX], validy),
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, tensorboard_callback, checkpoint],
                    verbose=1, class_weight=class_weight)

# summarize history for loss
plt.plot(history.history['prc'])
plt.plot(history.history['val_prc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('prc')
plt.xlabel('epoch')
plt.legend(['train prc', 'validation prc', 'train loss', 'validation loss'], loc='upper right')
plt.show()

# Saving model data
model_json = model.to_json()
with open("HNN_GoogleNet.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
