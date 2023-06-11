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
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/final_surv.csv')
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
json_file = open('models/mortality/Image_CNN/image_mortality_VGG19.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/mortality/Image_CNN/image_mortality_VGG19_my_model.best.hdf5")

# Predict with model
preds1 = model1.predict(testX)
pred_test_cl1 = pred_test_cl1 = list(map(lambda x: 0 if x[0]<0.5 else 1, preds1))
print(pred_test_cl1[:5])
survival_yhat = np.array(df.pop('Event_x'))
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
print('Image CNN score:',f1_score(survival_yhat, pred_test_cl1))

# Load trained mixed model
# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)_x','Essential_hypertension_x', 'Gender_x', 'Heart_failure_(disorder)_x', 'Smoking_history_x',
'Dyslipidaemia_x', 'Myocardial_infarction_(disorder)_x', 'Diabetes_mellitus_(disorder)_x', 'Cerebrovascular_accident_(disorder)_x']
numerical_col_list= ['Age_on_20.08.2021_x_x', 'LVEF_(%)_x']
testImageX = testX / 255.0

def process_attributes(df):
    continuous = numerical_col_list
    categorical = categorical_col_list
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(df[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    testCategorical = catBinarizer.transform(df[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    testX = np.hstack([testCategorical, testContinuous])

    return (testX)

testAttrX = process_attributes(df)
testAttrX = np.array(testAttrX)

# Load model
json_file = open('models/mortality/HNN/mixed_mortality_VGG19.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights('models/mortality/HNN/mixed_mortality_VGG19_my_model.best.hdf5')

# Predict with model
preds2 = model2.predict([testAttrX, testImageX])
pred_test_cl2 = pred_test_cl1 = list(map(lambda x: 0 if x[0]<0.5 else 1, preds2))
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
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_x', 'LVEF_(%)']

dirs1 = []
dirs2 = []
dir_1 = os.listdir('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/peak_LV_images')
dir_2 = os.listdir('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/peak_LV_test')
dirp = dir_2 + dir_1

dir_3 = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_img')
dir_4 = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_test')
dirl = dir_3 + dir_4

for d in dirp:
    if '.DS_Store' in dirp:
        dirp.remove('.DS_Store')
    folder_strip = d.rstrip('_')
    dirs1.append(int(folder_strip))

for d in dirl:
    if '.DS_Store' in dirl:
        dirl.remove('.DS_Store')
    folder_strip = d.rstrip('_')
    dirs2.append(int(folder_strip))

df1 = pd.DataFrame(dirs1, columns=['index'])
df1['ID'] = df1['index'].astype(int)
df2 = pd.DataFrame(dirs2, columns=['index'])
df2['ID'] = df2['index'].astype(int)
df = pd.merge(df1, df2, on=['ID'])

# Create dataframe
data = patient_df.merge(df, on=['ID'])
print(len(data))

# Loading clinical data
trainx, testx = utils.patient_dataset_splitter(data, patient_key='patient_TrustNumber')
y_train = np.array(trainx.pop('Event'))
y_test = np.array(testx.pop('Event'))
x_train = np.array(trainx[categorical_col_list + numerical_col_list])
x_test = np.array(testx[categorical_col_list + numerical_col_list])

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
plt.plot(fpr, tpr, label="Mixed NN , AUC="+str(auc))
fpr, tpr, _ = roc_curve(survival_yhat, preds1[:,0])
auc = round(roc_auc_score(survival_yhat, preds1[:,0]), 2)
plt.plot(fpr, tpr, label="Image CNN , AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, lr_preds)
auc = round(roc_auc_score(y_test, lr_preds), 2)
plt.plot(fpr,tpr,label="Clinical ML Model, AUC="+str(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('AUC Models Comparison')
plt.show()

# Plot PR
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds2[:,0])
label='%s (F1 Score:%0.2f)' % ('Mixed NN', average_precision_score(survival_yhat, preds2[:,0]))
plt.plot(recall, precision, label=label)
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds1[:,0])
label='%s (F1 Score:%0.2f)' % ('Image CNN', average_precision_score(survival_yhat, preds1[:,0]))
plt.plot(recall, precision, label=label)
precision, recall, thresholds = precision_recall_curve(y_test, lr_preds)
label='%s (F1 Score:%0.2f)' % ('Clinical ML Model', average_precision_score(y_test, lr_preds))
plt.plot(recall, precision, label=label)
plt.xlim(0.1, 1.2)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('F1 Score Models Comparison')
plt.show()
