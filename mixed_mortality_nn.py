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

class LeNet:
    @staticmethod
    def build_cnn(height, width, depth, regress=False):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(4))
        model.add(Activation("relu"))

        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))

        # return the constructed network architecture
        return model

class MiniVGGNet:
    @staticmethod
    def build(height, width, depth):
        # initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
		# and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 0

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(4))
        model.add(Activation("relu"))

        return model

class AlexNet:
    @staticmethod
    def build(height, width, depth, reg=0.0002):
        # initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
		# and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block #1: first CONV => RELU => POOL layer set
        model.add(Conv2D(96, (11, 11), strides=(4, 4),
			input_shape=inputShape, padding="same",
			kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(256, (5, 5), padding="same",
			kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3, 3), padding="same",
			kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding="same",
			kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same",
			kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #5: second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(4))
        model.add(Activation("relu"))

        return model

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
#resnet_model = ResNet50(include_top=False, input_shape=(1344, 224, 3), weights='imagenet')

transfer_layer = vgg_model.get_layer('block5_pool')
#transfer_layer = resnet_model.get_layer('conv5_block3_out')
vgg_model = Model(inputs = vgg_model.input, outputs = transfer_layer.output)
#resnet_model = Model(inputs = resnet_model.input, outputs = transfer_layer.output)

for layer in vgg_model.layers[0:17]:
    layer.trainable = False
#for layer in resnet_model.layers:
 #   layer.trainable = False
my_model = Sequential()
my_model.add(vgg_model)
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(4))
my_model.add(Activation("relu"))

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

#df['Ventricular_tachycardia'] = df['Ventricular_tachycardia_(disorder)_x'].astype('int')
#df['Ventricular_fibrillation'] = df['Ventricular_fibrillation_(disorder)_x'].astype('int')
#df['VT'] = df[['Ventricular_tachycardia','Ventricular_fibrillation']].apply(lambda x:'{}'.format(np.max(x)), axis=1)
#df['VT'] = df['VT'].astype(int)

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
#p_ind_train = df_train[df_train[args["target"]]==1].index.tolist()
#np_ind_train = df_train[df_train[args["target"]]==0].index.tolist()
#np_sample_train = sample(np_ind_train, len(p_ind_train))
#df_train = df_train.loc[p_ind_train + np_sample_train]
#trainy = np.array(df_train[args["target"]])
#p_ind_valid = df_valid[df_valid[args["target"]]==1].index.tolist()
#np_ind_valid = df_valid[df_valid[args["target"]]==0].index.tolist()
#np_sample_valid = sample(np_ind_valid, ((len(df_valid) - vlist.count(1)) // vlist.count(1))*len(p_ind_valid))
#df_valid = df_valid.loc[p_ind_valid + np_sample_valid]
#validy = np.array(df_valid[args["target"]])
#X_train1 = np.array([x1 for x1 in df_train['Perf']])
#X_train2 = np.array([x2 for x2 in df_train['LGE']])
#trainImages = np.hstack((X_train1, X_train2))

X_train = np.array([x for x in df_train['images']])
print(X_train.shape)
X_valid = np.array([x for x in df_valid['images']])
print(X_valid.shape)
print(trainy[:10])
print(validy[:10])

train_gen = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.1, zoom_range
                         =0.1, horizontal_flip=True, fill_mode="nearest")
trainImages = train_gen.flow(X_train, batch_size=1000)
trainImagesX = trainImages.next()

valid_gen = ImageDataGenerator()
validImages = valid_gen.flow(X_valid, batch_size=500)
validImagesX = validImages.next()

#trainAttrX = df_train[categorical_col_list + numerical_col_list]
#trainAttrX = np.array(trainAttrX)
#validAttrX = df_valid[categorical_col_list + numerical_col_list]
#validAttrX = np.array(validAttrX)
#scaler = StandardScaler()
#trainAttrX = scaler.fit_transform(trainAttrX)
#validAttrX = scaler.transform(validAttrX)
#trainAttrX = np.clip(trainAttrX, -5, 5)
#validAttrX = np.clip(validAttrX, -5, 5)
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
#ann_viz(model, view=True, filename='model_tree', title='HNN')

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
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
early_stopping = EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=40,
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
