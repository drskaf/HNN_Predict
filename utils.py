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


def load_perf_images(directory, df, im_size):
    Images = []
    indices = []

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
            if len(files) > 2:
                Avideo = []
                Mvideo = []
                Bvideo = []
                videoraw = []
                VidGroup = []
                Location = []
                for file in files:
                    if not file.startswith('.'):
                        dicom = pydicom.read_file(os.path.join(dir_path, file))
                        location = dicom.SliceLocation
                        img = dicom.pixel_array
                        img = centre_crop(img)
                        img = resize(img, (im_size, im_size))
                        VidGroup.append([img, location])
                        Location.append(location)
                        videoraw.append(dicom)

                for vid, loc in VidGroup:
                    if loc == np.max(Location):
                        Bvideo.append(vid)
                    elif loc == np.min(Location):
                        Avideo.append(vid)
                    else:
                        Mvideo.append(vid)
                video = Avideo + Mvideo + Bvideo
                test = {}
                keys = range(len(video))
                # Define series with AIF frames
                for k in keys:
                    # Calculate image sharpness
                    img = centre_crop(video[k])
                    gy, gx = np.gradient(img)
                    gnorm = np.sqrt(gx ** 2, gy ** 2)
                    sharp = np.average(gnorm)
                    test[k] = sharp
                aif_f = len(video) // 4
                aif_sharp = np.sum(list(test.values())[:aif_f])
                nonaif_sharp = np.sum(list(test.values())[aif_f + 1:2 * aif_f])
                if aif_sharp < nonaif_sharp // 2:
                    '''Work on series with AIF frames'''
                    # Initiate dictionaries for 4 groups of AIF, apical, mid and basal LV level
                    # both for pixel sum values and pixel peak values for each frame
                    aif_tot = {}
                    a_tot = {}
                    m_tot = {}
                    b_tot = {}
                    f = len(video) // 4
                    # Loop over the keys and calculate the sum and peak pixel values for each frame
                    for k in keys:
                        img = centre_crop(video[0][k])
                        sum = np.sum(img)
                        # Collect frames from the first group of slices
                        if k <= f:
                            aif_tot[k] = sum
                        # Collect frames from the second group of slices
                        elif k > f and k <= 2 * f:
                            a_tot[k] = sum
                        elif k > 2 * f and k <= 3 * f:
                            m_tot[k] = sum
                        # Collect frames from the third group of slices
                        else:
                            b_tot[k] = sum

                    aif_5max = {}
                    a_5max = {}
                    m_5max = {}
                    b_5max = {}
                    for k, v in aif_tot.items():
                        if k >= 5 and k < (f - 3):
                            value = aif_tot[k - 2] + aif_tot[k - 1] + aif_tot[k] + aif_tot[k + 1] + aif_tot[k + 2]
                            aif_5max[k] = value
                        else:
                            aif_5max[k] = 0
                    aif_max_value = list(aif_5max.values())
                    aif_max_key = [key for key, value in aif_5max.items() if value == np.max(aif_max_value)]
                    aif_l = aif_max_key.pop()
                    for k, v in a_tot.items():
                        if k >= 5 + f and k < (2 * f - 3):
                            value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                            a_5max[k] = value
                        else:
                            a_5max[k] = 0
                    a_max_value = list(a_5max.values())
                    a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
                    a_l = a_max_key.pop()
                    for k, v in m_tot.items():
                        if k >= 5 + 2 * f and k < (3 * f - 3):
                            value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k - 2]
                            m_5max[k] = value
                        else:
                            m_5max[k] = 0
                    m_max_value = list(m_5max.values())
                    m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
                    m_l = m_max_key.pop()
                    for k, v in b_tot.items():
                        if k >= 5 + 3 * f and k < (4 * f - 3):
                            value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                            b_5max[k] = value
                        else:
                            b_5max[k] = 0
                    b_max_value = list(b_5max.values())
                    b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
                    b_l = b_max_key.pop()

                    slice_dic = {videoraw[a_l].SliceLocation: video[a_l - 1:a_l + 3],
                                 videoraw[m_l].SliceLocation: video[m_l - 1:m_l + 3],
                                 videoraw[b_l].SliceLocation: video[b_l - 1:b_l + 3]}
                    sd = SortedDict(slice_dic)
                    if len(sd) == 3:
                        img1 = slice_dic[sd.iloc[0]]
                        img2 = slice_dic[sd.iloc[1]]
                        img3 = slice_dic[sd.iloc[2]]
                        img = np.stack([img1, img2, img3], axis=2)
                        Images.append(img)
                        #indices.append(folder_strip)
                    else:
                        img1 = video[al-1:a_l + 3]
                        img2 = video[ml-1:m_l + 3]
                        img3 = video[b_l-1:b_l + 3]
                        img = np.stack([img1, img2, img3], axis=2)
                        Images.append(img)
                        #indices.append(folder_strip)

                else:
                    '''Work on series without AIF frames'''
                    # Initiate dictionaries
                    a_tot = {}
                    m_tot = {}
                    b_tot = {}
                    f = len(video) // 3
                    for k in keys:
                        img = centre_crop(video[k])
                        sum = np.sum(img)
                        # Collect frames from the first group of slices
                        if k <= f:
                            a_tot[k] = sum
                        # Collect frames from the second group of slices
                        elif k > f and k <= 2 * f:
                            m_tot[k] = sum
                        # Collect frames from the third group of slices
                        else:
                            b_tot[k] = sum
                    # Generate a list of peak pixels values then find the key of the frame with the max value,
                    # this will be followed by 4-5 frames to get the myocardial contrast frame
                    # This will be done on all 3 groups of slices
                    # First, identify sequence which performs better with sum pixel rather than peak
                    a_5max = {}
                    m_5max = {}
                    b_5max = {}
                    # Working on 1st group
                    for k, v in a_tot.items():
                        if k >= 5 and k < (f - 3):
                            value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                            a_5max[k] = value
                        else:
                            a_5max[k] = 0
                    a_max_value = list(a_5max.values())
                    a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
                    a_l = a_max_key.pop()
                    # Working on 2nd group
                    for k, v in m_tot.items():
                        if k >= 3 + f and k < (2 * f - 3):
                            value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k + 2]
                            m_5max[k] = value
                        else:
                            m_5max[k] = 0
                    m_max_value = list(m_5max.values())
                    m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
                    m_l = m_max_key.pop()

                    # Working on 3rd group
                    for k, v in b_tot.items():
                        if k >= 5 + 2 * f and k < (3 * f - 3):
                            value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                            b_5max[k] = value
                        else:
                            b_5max[k] = 0
                    b_max_value = list(b_5max.values())
                    b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
                    b_l = b_max_key.pop()

                    slice_dic = {videoraw[a_l].SliceLocation: video[a_l - 1:a_l + 3],
                                 videoraw[m_l].SliceLocation: video[m_l - 1:m_l + 3],
                                 videoraw[b_l].SliceLocation: video[b_l - 1:b_l + 3]}
                    sd = SortedDict(slice_dic)
                    if len(sd) == 3:
                        img1 = slice_dic[sd.iloc[0]]
                        img2 = slice_dic[sd.iloc[1]]
                        img3 = slice_dic[sd.iloc[2]]
                        img = img1 + img2 + img3
                        img = np.stack(img, axis=2)
                        Images.append(img)
                        #indices.append(folder_strip)
                    else:
                        img1 = video[a_l - 1:a_l + 3]
                        img2 = video[m_l - 1:m_l + 3]
                        img3 = video[b_l - 1:b_l + 3]
                        img = img1 + img2 + img3
                        img = np.stack(img, axis=2)
                        Images.append(img)
                        #indices.append(folder_strip)

            else:
                """ EXTRACT PEAK LV frames from stacked dicoms"""
                for file in files:
                    if not file.startswith('.'):
                        videoraw = pydicom.read_file(os.path.join(dir_path, file))
                        video = videoraw.pixel_array

                        test = {}
                        keys = range(len(video[:,]))
                        # Define series with AIF frames
                        for k in keys:
                            # Calculate image sharpness
                            img = centre_crop(video[k])
                            gy, gx = np.gradient(img)
                            gnorm = np.sqrt(gx ** 2, gy ** 2)
                            sharp = np.average(gnorm)
                            test[k] = sharp
                        aif_f = len(video[:,]) // 4
                        aif_sharp = np.sum(list(test.values())[:aif_f])
                        nonaif_sharp = np.sum(list(test.values())[aif_f + 1:2 * aif_f])
                        if aif_sharp < nonaif_sharp // 2:
                            '''Work on series with AIF frames'''
                            # Initiate dictionaries for 4 groups of AIF, apical, mid and basal LV level
                            # both for pixel sum values and pixel peak values for each frame
                            aif_tot = {}
                            a_tot = {}
                            m_tot = {}
                            b_tot = {}
                            f = len(video[:,]) // 4
                            # Loop over the keys and calculate the sum and peak pixel values for each frame
                            for k in keys:
                                img = centre_crop(video[k])
                                sum = np.sum(img)
                                # Collect frames from the first group of slices
                                if k <= f:
                                    aif_tot[k] = sum
                                # Collect frames from the second group of slices
                                elif k > f and k <= 2 * f:
                                    a_tot[k] = sum
                                elif k > 2 * f and k <= 3 * f:
                                    m_tot[k] = sum
                                # Collect frames from the third group of slices
                                else:
                                    b_tot[k] = sum

                            aif_5max = {}
                            a_5max = {}
                            m_5max = {}
                            b_5max = {}
                            for k, v in aif_tot.items():
                                if k >= 5 and k < (f - 3):
                                    value = aif_tot[k - 2] + aif_tot[k - 1] + aif_tot[k] + aif_tot[k + 1] + aif_tot[
                                        k + 2]
                                    aif_5max[k] = value
                                else:
                                    aif_5max[k] = 0
                            aif_max_value = list(aif_5max.values())
                            aif_max_key = [key for key, value in aif_5max.items() if value == np.max(aif_max_value)]
                            aif_l = aif_max_key.pop()
                            for k, v in a_tot.items():
                                if k >= 5 + f and k < (2 * f - 3):
                                    value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                                    a_5max[k] = value
                                else:
                                    a_5max[k] = 0
                            a_max_value = list(a_5max.values())
                            a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
                            a_l = a_max_key.pop()
                            for k, v in m_tot.items():
                                if k >= 5 + 2 * f and k < (3 * f - 3):
                                    value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k - 2]
                                    m_5max[k] = value
                                else:
                                    m_5max[k] = 0
                            m_max_value = list(m_5max.values())
                            m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
                            m_l = m_max_key.pop()
                            for k, v in b_tot.items():
                                if k >= 5 + 3 * f and k < (4 * f - 3):
                                    value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                                    b_5max[k] = value
                                else:
                                    b_5max[k] = 0
                            b_max_value = list(b_5max.values())
                            b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
                            b_l = b_max_key.pop()

                            if 'SliceLocation' in videoraw:
                                slice_dic = {videoraw[a_l].SliceLocation: video[a_l - 1:a_l + 3],
                                             videoraw[m_l].SliceLocation: video[m_l - 1:m_l + 3],
                                             videoraw[b_l].SliceLocation: video[b_l - 1:b_l + 3]}
                                sd = SortedDict(slice_dic)
                                if len(sd) == 3:
                                    img1 = []
                                    for m in slice_dic[sd.iloc[0]]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img1.append(m)
                                    img2 = []
                                    for m in slice_dic[sd.iloc[1]]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img2.append(m)
                                    img3 = []
                                    for m in slice_dic[sd.iloc[2]]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img3.append(m)
                                    img = img1 + img2 + img3
                                    img = np.stack(img, axis=2)
                                    Images.append(img)
                                    #indices.append(folder_strip)
                                else:
                                    img1 = []
                                    for m in video[a_l - 1:a_l + 3]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img1.append(m)
                                    img2 = []
                                    for m in video[m_l - 1:m_l + 3]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img2.append(m)
                                    img3 = []
                                    for m in video[b_l - 1:b_l + 3]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img3.append(m)
                                    img = img1 + img2 + img3
                                    img = np.stack(img, axis=2)
                                    Images.append(img)
                                    #indices.append(folder_strip)

                            else:
                                img1 = []
                                for m in video[a_l - 1:a_l + 3]:
                                    m = centre_crop(m)
                                    m = resize(m, (im_size, im_size))
                                    img1.append(m)
                                img2 = []
                                for m in video[m_l - 1:m_l + 3]:
                                    m = centre_crop(m)
                                    m = resize(m, (im_size, im_size))
                                    img2.append(m)
                                img3 = []
                                for m in video[b_l - 1:b_l + 3]:
                                    m = centre_crop(m)
                                    m = resize(m, (im_size, im_size))
                                    img3.append(m)
                                img = img1 + img2 + img3
                                img = np.stack(img, axis=2)
                                Images.append(img)
                                #indices.append(folder_strip)

                        else:
                            '''Work on series without AIF frames'''
                            # Initiate dictionaries
                            a_tot = {}
                            m_tot = {}
                            b_tot = {}
                            f = len(video) // 3
                            for k in keys:
                                img = centre_crop(video[k])
                                sum = np.sum(img)
                                # Collect frames from the first group of slices
                                if k <= f:
                                    a_tot[k] = sum
                                # Collect frames from the second group of slices
                                elif k > f and k <= 2 * f:
                                    m_tot[k] = sum
                                # Collect frames from the third group of slices
                                else:
                                    b_tot[k] = sum
                            # Generate a list of peak pixels values then find the key of the frame with the max value,
                            # this will be followed by 4-5 frames to get the myocardial contrast frame
                            # This will be done on all 3 groups of slices
                            # First, identify sequence which performs better with sum pixel rather than peak
                            a_5max = {}
                            m_5max = {}
                            b_5max = {}
                            # Working on 1st group
                            for k, v in a_tot.items():
                                if k >= 5 and k < (f - 3):
                                    value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                                    a_5max[k] = value
                                else:
                                    a_5max[k] = 0
                            a_max_value = list(a_5max.values())
                            a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
                            a_l = a_max_key.pop()
                            # Working on 2nd group
                            for k, v in m_tot.items():
                                if k >= 3 + f and k < (2 * f - 3):
                                    value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k + 2]
                                    m_5max[k] = value
                                else:
                                    m_5max[k] = 0
                            m_max_value = list(m_5max.values())
                            m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
                            m_l = m_max_key.pop()

                            # Working on 3rd group
                            for k, v in b_tot.items():
                                if k >= 5 + 2 * f and k < (3 * f - 3):
                                    value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                                    b_5max[k] = value
                                else:
                                    b_5max[k] = 0
                            b_max_value = list(b_5max.values())
                            b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
                            b_l = b_max_key.pop()

                            if 'SliceLocation' in videoraw:
                                slice_dic = {videoraw[a_l].SliceLocation: video[a_l - 1:a_l + 3],
                                             videoraw[m_l].SliceLocation: video[m_l - 1:m_l + 3],
                                             videoraw[b_l].SliceLocation: video[b_l - 1:b_l + 3]}
                                sd = SortedDict(slice_dic)
                                if len(sd) == 3:
                                    img1 = []
                                    for m in slice_dic[sd.iloc[0]]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img1.append(m)
                                    img2 = []
                                    for m in slice_dic[sd.iloc[1]]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img2.append(m)
                                    img3 = []
                                    for m in slice_dic[sd.iloc[2]]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img3.append(m)
                                    img = img1 + img2 + img3
                                    img = np.stack(img, axis=2)
                                    Images.append(img)
                                    #indices.append(folder_strip)
                                else:
                                    img1 = []
                                    for m in video[a_l - 1:a_l + 3]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img1.append(m)
                                    img2 = []
                                    for m in video[m_l - 1:m_l + 3]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img2.append(m)
                                    img3 = []
                                    for m in video[b_l - 1:b_l + 3]:
                                        m = centre_crop(m)
                                        m = resize(m, (im_size, im_size))
                                        img3.append(m)
                                    img = img1 + img2 + img3
                                    img = np.stack(img, axis=2)
                                    Images.append(img)
                                    #indices.append(folder_strip)

                            else:
                                img1 = []
                                for m in video[a_l - 1:a_l + 3]:
                                    m = centre_crop(m)
                                    m = resize(m, (im_size, im_size))
                                    img1.append(m)
                                img2 = []
                                for m in video[m_l - 1:m_l + 3]:
                                    m = centre_crop(m)
                                    m = resize(m, (im_size, im_size))
                                    img2.append(m)
                                img3 = []
                                for m in video[b_l - 1:b_l + 3]:
                                    m = centre_crop(m)
                                    m = resize(m, (im_size, im_size))
                                    img3.append(m)
                                img = img1 + img2 + img3
                                img = np.stack(img, axis=2)
                                Images.append(img)
            indices.append(int(folder_strip))

    idx_df = pd.DataFrame(indices, columns=['ID'])
    idx_df['Perf'] = Images
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

