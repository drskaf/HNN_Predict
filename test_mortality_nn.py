import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score
import pickle
from random import sample
from sklearn.model_selection import train_test_split
import utils
from sklearn.preprocessing import StandardScaler
import tempfile

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
patient_df['Gender'] = patient_df['patient_GenderCode_x'].astype('category')
patient_df['Gender'] = patient_df['Gender'].cat.codes

# Load images
(df1) = utils.load_label_png('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/peak_LV_test', patient_df, 224)
(df2) = utils.load_lge_data('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_test', patient_df, 224)
df = df1.merge(df2, on='ID')
print(len(df))
X_test1 = np.array([x1 for x1 in df['Perf']])
X_test2 = np.array([x2 for x2 in df['LGE']])
testX = np.hstack((X_test1, X_test2))

# Load trained image model
json_file = open('models/VA/image_VA_VGG19.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/VA/image_VA_VGG19_my_model.best.hdf5")

# Predict with model
preds1 = model1.predict(testX)
pred_test_cl1 = list(map(lambda x: 0 if x[0]<0.5 else 1, preds1))
print(pred_test_cl1[:5])
survival_yhat = np.array(df.pop('VT'))
print(survival_yhat[:5])

prob_outputs1 = {
    "pred": pred_test_cl1,
    "actual_value": survival_yhat
}
prob_output_df1 = pd.DataFrame(prob_outputs1)
print(prob_output_df1.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl1))
print('Image CNN ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl1))
print('Image CNN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl1))
print('Image CNN F1 score:',f1_score(survival_yhat, pred_test_cl1))

# Load trained mixed model
# Define columns
categorical_col_listm = ['Chronic_kidney_disease_(disorder)_x','Essential_hypertension_x', 'Gender_x', 'Heart_failure_(disorder)_x', 'Smoking_history_x',
'Dyslipidaemia_x', 'Myocardial_infarction_(disorder)_x', 'Diabetes_mellitus_(disorder)_x', 'Cerebrovascular_accident_(disorder)_x']
numerical_col_listm= ['Age_on_20.08.2021_x_x', 'LVEF_(%)_x']

def process_attributes(df):
    continuous = numerical_col_listm
    categorical = categorical_col_listm
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(df[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    testCategorical = catBinarizer.transform(df[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    testX = np.hstack([testCategorical, testContinuous])

    return (testX)

(df1) = utils.load_label_png('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/peak_LV_test', patient_df, 224)
(df2) = utils.load_lge_data('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_test', patient_df, 224)
df = df1.merge(df2, on='ID')
print(len(df))
X_test1 = np.array([x1 for x1 in df['Perf']])
X_test2 = np.array([x2 for x2 in df['LGE']])
testImageX = testX / 255.0 #np.hstack((X_test1, X_test2)) / 255.0

testAttrX = process_attributes(df)
testAttrX = np.array(testAttrX)

# Load model
json_file = open('models/VA/mixed_VA_VGG19.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights('models/VA/mixed_VA_VGG19_my_model.best.hdf5')

# Predict with model
preds2 = model2.predict([testAttrX, testImageX])
pred_test_cl2 = list(map(lambda x: 0 if x[0]<0.5 else 1, preds2))
print(pred_test_cl2[:5])

prob_outputs2 = {
    "pred": pred_test_cl2,
    "actual_value": survival_yhat
}
prob_output_df2 = pd.DataFrame(prob_outputs2)
print(prob_output_df2.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl2))
print('HNN ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl2))
print('HNN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl2))
print('HNN F1 score:',f1_score(survival_yhat, pred_test_cl2))

# Train ML model on clinical data
# Define data file

trainx, testx = utils.patient_dataset_splitter(patient_df, patient_key='patient_TrustNumber')
y_train = np.array(trainx.pop('VT'))
y_test = np.array(testx.pop('VT'))
x_train = np.array(process_attributes(trainx))
x_test = np.array(process_attributes(testx))

# fit Linear model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
lr_predict = lr_model.predict(x_test)
lr_preds = lr_model.predict_proba(x_test)[:,1]

print('Linear ROCAUC score:',roc_auc_score(y_test, lr_preds))
print('Linear Accuracy score:',accuracy_score(y_test, lr_predict))
print('Linear F1 score:',f1_score(y_test, lr_predict))

# Plot ROC
fpr, tpr, _ = roc_curve(survival_yhat, preds2[:,0])
auc = round(roc_auc_score(survival_yhat, preds2[:,0]), 2)
plt.plot(fpr, tpr, label="HNN AUC="+str(auc), color='purple')
fpr, tpr, _ = roc_curve(survival_yhat, preds1[:,0])
auc = round(roc_auc_score(survival_yhat, preds1[:,0]), 2)
plt.plot(fpr, tpr, label="Image CNN , AUC="+str(auc), color='navy')
fpr, tpr, _ = roc_curve(y_test, lr_preds)
auc = round(roc_auc_score(y_test, lr_preds), 2)
plt.plot(fpr,tpr,label="Clinical ML Model, AUC="+str(auc), color='gold')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('AUC Models Comparison')
plt.grid()
plt.show()

# Plot PR
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds2[:,0])
label='%s (F1 Score:%0.2f)' % ('HNN', average_precision_score(survival_yhat, preds2[:,0]))
plt.plot(recall, precision, label=label, color='purple')
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds1[:,0])
label='%s (F1 Score:%0.2f)' % ('Image CNN', average_precision_score(survival_yhat, preds1[:,0]))
plt.plot(recall, precision, label=label, color='navy')
precision, recall, thresholds = precision_recall_curve(y_test, lr_preds)
label='%s (F1 Score:%0.2f)' % ('Clinical ML Model', average_precision_score(y_test, lr_preds))
plt.plot(recall, precision, label=label, color='gold')
plt.xlim(0.1, 1.2)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('F1 Score Models Comparison')
plt.grid()
plt.show()

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

print('DeLong test for non-linear and linear predictions:', delong_roc_test(y_test, preds1[:,0], preds2[:,0]))
