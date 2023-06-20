import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import cv2
import os
import argparse
import utils
import matplotlib.image as mpimg
from skimage.transform import resize
from keras.models import Sequential
from keras. layers import Input, Dense, Flatten, concatenate, Conv2D, Activation, MaxPool2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras import backend as K
from datetime import datetime
from sklearn.preprocessing import StandardScaler
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
HEIGHT = 1344
BATCH_SIZE = 32
NUM_EPOCHS = 500
STEP_PER_EPOCH = 50
N_CLASSES = 1

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
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

from keras.applications.vgg16 import VGG16
vgg_model = VGG16(include_top=False, input_shape=(1344, 224, 3), weights='imagenet')

transfer_layer = vgg_model.get_layer('block5_pool')
vgg_model = Model(inputs = vgg_model.input, outputs = transfer_layer.output)

for layer in vgg_model.layers[0:17]:
    layer.trainable = False
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
(df1) = utils.load_label_png('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/peak_LV_images', patient_df, INPUT_DIM)
print(len(df1))
(df2) = utils.load_lge_data('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_img', patient_df, INPUT_DIM)
print(len(df2))
df = df1.merge(df2, on='ID')
print(len(df))

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
p_ind_train = df_train[df_train[args["target"]]==1].index.tolist()
np_ind_train = df_train[df_train[args["target"]]==0].index.tolist()
np_sample_train = sample(np_ind_train, len(p_ind_train))
df_train = df_train.loc[p_ind_train + np_sample_train]
p_ind_valid = df_valid[df_valid[args["target"]]==1].index.tolist()
np_ind_valid = df_valid[df_valid[args["target"]]==0].index.tolist()
np_sample_valid = sample(np_ind_valid, 11*len(p_ind_valid))
df_valid = df_valid.loc[p_ind_valid + np_sample_valid]
X_train1 = np.array([x1 for x1 in df_train['Perf']])
X_train2 = np.array([x2 for x2 in df_train['LGE']])
trainImages = np.hstack((X_train1, X_train2))
train_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")
trainImages = train_gen.flow(trainImages, batch_size=2000)
trainImagesX = trainImages.next()

X_valid1 = np.array([x1 for x1 in df_valid['Perf']])
X_valid2 = np.array([x2  for x2 in df_valid['LGE']])
validImages = np.hstack((X_valid1, X_valid2))
valid_gen = v_aug = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
validImages = valid_gen.flow(validImages, batch_size=1000)
validImagesX = validImages.next()

(trainAttrX, validAttrX) = process_attributes(df, df_train, df_valid)

# find the targest field in the training set
trainy = np.array(df_train.pop(args["target"]))
tlist = trainy.tolist()
print(tlist.count(1))
validy = np.array(df_valid.pop(args["target"]))
vlist = validy.tolist()
print(vlist.count(1))

# create the MLP and CNN models
mlp = create_mlp(trainAttrX.shape[1], regress=False)
cnn = my_model #AlexNet.build(HEIGHT, WIDTH, 1)
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
weigth_path = "{}_my_model.best.hdf5".format("mixed_VA_VGG19")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)
early_stopping = EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)
callback = LearningRateScheduler(scheduler)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# train the model
print("[INFO] training model...")
history = model.fit(x=[trainAttrX, trainImagesX], y=trainy, validation_data=([validAttrX, validImagesX], validy),
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, tensorboard_callback, checkpoint], verbose=1)

# summarize history for loss
plt.plot(history.history['prc'])
plt.plot(history.history['val_prc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('prc_loss')
plt.xlabel('epoch')
plt.legend(['train prc', 'validation prc', 'train loss', 'validation loss'], loc='upper right')
plt.show()

# Saving model data
model_json = model.to_json()
with open("mixed_VA_VGG19.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
