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
from keras. layers import Input, Dense, Flatten, concatenate, Conv2D, Activation, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical
from datetime import datetime


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 64
NUM_EPOCHS = 250
STEP_PER_EPOCH = 50
N_CLASSES = 2

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_x', 'LVEF_(%)']

# Loading clinical data
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

# Loading images
def load_images(directory, im_size):
    # initialize our images array
    images = []
    indices = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img1 = mpimg.imread(os.path.join(dir_path, file))
                    img1 = resize(img1, (im_size, im_size))
                if file_name == 'm':
                    img2 = mpimg.imread(os.path.join(dir_path, file))
                    img2 = resize(img2, (im_size, im_size))
                if file_name == 'a':
                    img3 = mpimg.imread(os.path.join(dir_path, file))
                    img3 = resize(img3, (im_size, im_size))

                    out = cv2.vconcat([img1, img2, img3])
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    out = gray[..., np.newaxis]

                    indices.append(int(folder))
                    images.append(out)

    return (np.array(images), indices)

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
    def build_cnn(width, height, depth, regress=False):
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

images, indices = load_images(args["directory"], im_size=224)
images = images / 255.0
df = pd.DataFrame(indices, columns=['ID'])
info_df = pd.merge(df, patient_df, on=['ID'])

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
print("[INFO] processing data...")
split = train_test_split(info_df, images, test_size=0.20, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
# find the largest field in the training set
trainy = trainAttrX.pop(args["target"])
le = LabelEncoder().fit(trainy)
trainY = to_categorical(le.transform(trainy), 2)
testy = testAttrX.pop(args["target"])
le = LabelEncoder().fit(testy)
testY = to_categorical(le.transform(testy), 2)
# process the clinical data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = process_attributes(info_df,
	trainAttrX, testAttrX)

# create the MLP and CNN models
mlp = create_mlp(trainAttrX.shape[1], regress=False)
cnn = LeNet.build_cnn(WIDTH, HEIGHT, 1, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(N_CLASSES, activation="softmax")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (outcome
# prediction)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using binary categorical cross-entropy given that
# we have binary classes of either the prediction is positive or negative
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
weigth_path = "{}_my_model.best.hdf5".format("#mixed_mortality_predictor")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
callbacks_list = [checkpoint]
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# train the model
print("[INFO] training model...")
history = model.fit(x=[trainAttrX, trainImagesX], y=trainY, validation_data=([testAttrX, testImagesX], testY),
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[callbacks_list, tensorboard_callback])

# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='upper left')
plt.show()

# Saving model data
image_model.save("image_mortality_predictor")
model_json = image_model.to_json()
with open("image_mixedmodel.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
