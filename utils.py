import imageio.v2
import numpy as np
import pydicom
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
import glob
from collections import Counter
import tensorflow as tf
import functools
import dcmstack
import dicom
import imageio
import dicom2nifti
from pathlib import Path
import nibabel as nib
import dicom2nifti.settings as settings
import tqdm
import highdicom
from natsort import natsorted
import ipyplot
from PIL import Image
import re


def load_all_data(directory1, directory2, df, im_size):
    """
    Args:
     directory: the path to the folder where images are stored
    Return:
        list of images and indices
    """

    images = []
    indices = []

    dirs1 = os.listdir(directory1)
    dirs2 = os.listdir(directory2)
    dirsCom = set(dirs1).intersection(dirs2)
    # Loop over folders and files
    for root, dirs, files in os.walk(directory1, topdown=True):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for dirp in dirs:
            if dirp in dirsCom:
                folder_strip = dirp.rstrip('_')
                imgList = []
                dir_path = os.path.join(directory1, dirp)
                files = sorted(os.listdir(dir_path))
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                for file in files:
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    imgList.append(img)
                for root, dirs, files in os.walk(directory2, topdown=True):
                    if '.DS_Store' in files:
                        files.remove('.DS_Store')
                    for dirl in dirs:
                        if dirl == dirp:
                            dir_path = os.path.join(directory2, dirl)
                            files = sorted(os.listdir(dir_path))
                            if '.DS_Store' in files:
                                files.remove('.DS_Store')
                            for file in files:
                                img = mpimg.imread(os.path.join(dir_path, file))
                                img = resize(img, (im_size, im_size))
                                imgList.append(img)
                        else:
                            continue

                images.append(imgList)
                indices.append(int(folder_strip))

            else:
                continue
    Images = []
    for image_list in images:
        img = cv2.vconcat(image_list)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = resize(gray, (3584, 224))
        out = cv2.merge([gray, gray, gray])
        # out = gray[..., np.newaxis]
        Images.append(out)

    idx_df = pd.DataFrame(indices, columns=['ID'])
    idx_df['images'] = Images
    info_df = pd.merge(df, idx_df, on=['ID'])

    return (info_df)

def patient_dataset_splitter(df, patient_key='patient_TrustNumber'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
    '''

    df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    train_size = round(total_values * 0.8)
    train = df[df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[train_size:])].reset_index(drop=True)

    return train, validation


def convertNsave(arr, file_dir, dcm_path, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    """

    dicom_file = pydicom.read_file(dcm_path)
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))

def nifti2dicom_1file(nifti_dir, out_dir, dcm_path):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[0]

    multi_dicom = []
    for slice_ in range(number_slices):
        dicom = convertNsave(nifti_array[slice_,:,:], out_dir, dcm_path, slice_)
        multi_dicom.append(dicom)

    return multi_dicom


def load_perfusion_data(directory):
    """
    Args:
     directory: the path to the folder where dicom images are stored
    Return:
        combined 3D files with 1st dimension as frames depth
    """

    videoStackList = []
    indicesStackList = []
    videoSingleList = []
    indicesSingleList = []
    videorawList = []

    dir_paths = sorted(glob.glob(os.path.join(directory, "*")))
    for dir_path in dir_paths:
        file_paths = sorted(glob.glob(os.path.join(dir_path, "*.dcm")))

        if len(file_paths) > 10:
            folder = os.path.split(dir_path)[1]
            print("\nWorking on ", folder)
            vlist = []
            vrlist = []
            for file_path in file_paths:
                imgraw = pydicom.read_file(file_path)
                vrlist.append(imgraw)
                img = imgraw.pixel_array
                vlist.append(img)
            videorawList.append(vrlist)
            videoSingleList.append(vlist)
            indicesSingleList.append(folder)

        else:
            folder = os.path.split(dir_path)[1]
            print("\nWorking on ", folder)
            for i in file_paths[0:]:
                # Read stacked dicom and add to list
                videoraw = pydicom.read_file(os.path.join(dir_path, i), force=True)
                videoStackList.append(videoraw)
                indicesStackList.append(folder)

    return videorawList, videoSingleList, indicesSingleList, videoStackList, indicesStackList


def balance_data(df, target_size=12):
    """
    Increase the number of samples to number_of_samples for every label

        Example:
        Current size of the label a: 10
        Target size: 23

        repeat, mod = divmod(target_size,current_size)
        2, 3 = divmod(23,10)

        Target size: current size * repeat + mod

    Repeat this example for every label in the dataset.
    """

    df_groups = df.groupby(['label'])
    df_balanced = pd.DataFrame({key: [] for key in df.keys()})

    for i in df_groups.groups.keys():
        df_group = df_groups.get_group(i)
        df_label = df_group.sample(frac=1)
        current_size = len(df_label)

        if current_size >= target_size:
            # If current size is big enough, do nothing
            pass
        else:

            # Repeat the current dataset if it is smaller than target_size
            repeat, mod = divmod(target_size, current_size)

            df_label_new = pd.concat([df_label] * repeat, ignore_index=True, axis=0)
            df_label_remainder = df_group.sample(n=mod)

            df_label_new = pd.concat([df_label_new, df_label_remainder], ignore_index=True, axis=0)

            # print(df_label_new)

        df_balanced = pd.concat([df_balanced, df_label_new], ignore_index=True, axis=0)

    return df_balanced

def centre_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = width//1.5

    if new_height is None:
        new_height = height//1.5

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        centre_cropped_img = img[top:bottom, left:right]
    else:
        centre_cropped_img = img[top:bottom, left:right, ...]

    return centre_cropped_img


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

