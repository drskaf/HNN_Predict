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


def load_perf_data(directory, df, im_size):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    #labels = []
    indices = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img1 = mpimg.imread(os.path.join(dir_path, file))
                    img1 = resize(img1, (im_size, im_size))
                elif file_name == 'm':
                    img2 = mpimg.imread(os.path.join(dir_path, file))
                    img2 = resize(img2, (im_size, im_size))
                elif file_name == 'a':
                    img3 = mpimg.imread(os.path.join(dir_path, file))
                    img3 = resize(img3, (im_size, im_size))

                    out = cv2.vconcat([img1, img2, img3])
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (672, 224))
                    out = cv2.merge([gray, gray, gray])
                    #out = gray[..., np.newaxis]

                    images.append(out)
                    indices.append(int(folder_strip))

    idx_df = pd.DataFrame(indices, columns=['ID'])
    idx_df['Perf'] = images
    info_df = pd.merge(df, idx_df, on=['ID'])

    return (info_df)


def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def load_lge_images(directory, df, im_size):
    Images = []
    indices = []
    # outliers = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for dir in dirs:
            folder_strip = dir.rstrip('_')
            dir_path = os.path.join(directory, dir)
            files = sorted(os.listdir(dir_path))
            print("\nWorking on ", folder_strip)
            # Loop over cases with single dicoms
            if len(files) > 5:
                imgList_2ch = []
                imgList_3ch = []
                imgList_4ch = []
                bun_2c = []
                bun_3c = []
                bun_4c = []
                bun_sa = []
                for file in files:
                    if not file.startswith('.'):
                        dicom = pydicom.read_file(os.path.join(dir_path, file))
                        dicom_series = dicom.SeriesDescription
                        dicom_series = dicom_series.split('_')
                        #print(dicom_series)
                        img = dicom.pixel_array
                        img = resize(img, (im_size, im_size))
                        mat_2c = findWholeWord('2ch')(str(dicom_series))
                        mat_3c = findWholeWord('3ch')(str(dicom_series))
                        mat_4c = findWholeWord('4ch')(str(dicom_series))
                        sa = []

                        if mat_2c:
                            bun_2c.append(img)
                        elif mat_3c:
                            bun_3c.append(img)
                        elif mat_4c:
                            bun_4c.append(img)
                        else:
                            sa.append(img)
                        bun_sa.append(np.squeeze(sa))

                l = len(bun_sa) // 3
                imgList_sa = (bun_sa[l:l+10] if len(bun_sa) > 25 else bun_sa[1:11])
                imgList_2ch.append(bun_2c[0]) if len(bun_2c) == 1 else imgList_2ch.append(bun_2c[1])
                imgList_3ch.append(bun_3c[0]) if len(bun_3c) == 1 else imgList_3ch.append(bun_3c[1])
                imgList_4ch.append(bun_4c[0]) if len(bun_4c) == 1 else imgList_4ch.append(bun_4c[1])
                imgList = imgList_sa + imgList_2ch + imgList_3ch + imgList_4ch
                imgStack = np.stack(imgList, axis=2)
                Images.append(imgStack)

            # Loop over cases with stacked dicoms
            else:
                bun_2c = []
                bun_3c = []
                bun_4c = []
                bun_sa = []
                for file in files:
                    if not file.startswith('.'):
                        dicom = pydicom.read_file(os.path.join(dir_path, file))
                        dicom_series = dicom.SeriesDescription
                        dicom_series = dicom_series.split('_')
                        #print(dicom_series)
                        img = dicom.pixel_array
                        mat_2c = findWholeWord('2ch')(str(dicom_series))
                        mat_3c = findWholeWord('3ch')(str(dicom_series))
                        mat_4c = findWholeWord('4ch')(str(dicom_series))
                        #sa = []

                        if mat_2c:
                            images = range(len(img[:, ]))
                            if len(images) > 1:
                                img = img[1]
                                img = resize(img, (im_size, im_size))
                                bun_2c.append(img)
                            else:
                                img = img[0]
                                img = resize(img, (im_size, im_size))
                                bun_2c.append(img)
                        elif mat_3c:
                            images = range(len(img[:, ]))
                            if len(images) > 1:
                                img = img[1]
                                img = resize(img, (im_size, im_size))
                                bun_3c.append(img)
                            else:
                                img = img[0]
                                img = resize(img, (im_size, im_size))
                                bun_3c.append(img)
                        elif mat_4c:
                            images = range(len(img[:, ]))
                            if len(images) > 1:
                                img = img[1]
                                img = resize(img, (im_size, im_size))
                                bun_4c.append(img)
                            else:
                                img = img[0]
                                img = resize(img, (im_size, im_size))
                                bun_4c.append(img)
                        else:
                            images = range(len(img[:, ]))
                            print(len(images))
                            l = len(images) // 3
                            if len(images) > 25:
                                img = img[l:l+10]
                                for i in img[:]:
                                    img = resize(i, (im_size, im_size))
                                    bun_sa.append(img)
                            else:
                                img = img[1:11]
                                for i in img[:]:
                                    img = resize(i, (im_size, im_size))
                                    bun_sa.append(img)

                imgList = bun_sa + bun_2c + bun_3c + bun_4c
                imgStack = np.stack(imgList, axis=2)
                #print(imgStack.shape)
                Images.append(imgStack)
                
            indices.append(int(folder_strip))

    idx_df = pd.DataFrame(indices, columns=['ID'])
    idx_df['LGE'] = Images
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

