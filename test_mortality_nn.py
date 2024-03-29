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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score, precision_score, recall_score
import pickle
from random import sample
from sklearn.model_selection import train_test_split
import utils
from sklearn.preprocessing import StandardScaler
import tempfile
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import mcnemar_table, mcnemar
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/final_surv.csv')
patient_df['Gender'] = patient_df['patient_GenderCode_x'].astype('category')
patient_df['Gender'] = patient_df['Gender'].cat.codes

# Load images
(df1) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_test', patient_df, 224)
(df2) = utils.load_lge_images('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_test', patient_df, 224)
df = df1.merge(df2, on='ID')
print(len(df))
perf_imgs = np.array([x for x in df['Perf']])
lge_imgs = np.array([x for x in df['LGE']])
Imgs = []
for p, l in zip(perf_imgs, lge_imgs):
    i = np.block([p,l])
    Imgs.append(i)
df['images'] = Imgs
testX = np.array([x for x in df['images']])
print(testX.shape)

# Load trained image model
json_file = open('models/Image_CNN/image_AlexNet.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/Image_CNN/image_AlexNet_my_model.best.hdf5")

# Predict with model
preds1 = model1.predict(testX)
print(preds1[:5])
survival_yhat = np.array(df['Event_x'])
print(survival_yhat[:5])
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds1[:,0])
pred_test_cl1 = np.array(list(map(lambda x: 0 if x<np.mean(thresholds) else 1, preds1)))
print(pred_test_cl1[:5])

prob_outputs1 = {
    "pred": pred_test_cl1,
    "actual_value": survival_yhat
}
prob_output_df1 = pd.DataFrame(prob_outputs1)
print(prob_output_df1.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl1))
print('Image CNN ROCAUC score:',roc_auc_score(survival_yhat, preds1[:,0]))
print('Image CNN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl1))
print('Image CNN Precision:', np.mean(precision))
print('Image CNN recall:', np.mean(recall))
print('Image CNN F1 Score:',average_precision_score(survival_yhat, preds1[:,0]))

# plot confusion matrix
cm = confusion_matrix(survival_yhat, pred_test_cl1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

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

testImageX = testX

testAttrX = process_attributes(df)
testAttrX = np.array(testAttrX)

# Load model
json_file = open('models/HNN/HNN_GoogleNet.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights('models/HNN/HNN_GoogleNet_my_model.best.hdf5')

# Predict with model
preds2 = model2.predict([testAttrX, testImageX])
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds2[:,0])
pred_test_cl2 = np.array(list(map(lambda x: 0 if x<np.mean(thresholds) else 1, preds2)))
print(pred_test_cl2[:5])

prob_outputs2 = {
    "pred": pred_test_cl2,
    "actual_value": survival_yhat
}

prob_output_df2 = pd.DataFrame(prob_outputs2)
print(prob_output_df2.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl2))
print('HNN ROCAUC score:',roc_auc_score(survival_yhat, preds2[:,0]))
print('HNN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl2))
print('HNN Precision:', np.mean(precision))
print('HNN recall:', np.mean(recall))
print('HNN F1 Score:',average_precision_score(survival_yhat, preds2[:,0]))

# plot confusion matrix
cm = confusion_matrix(survival_yhat, pred_test_cl2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Train ML model on clinical data
# Define data file
#patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
categorical_col_listc = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_listc= ['Age_on_20.08.2021_x', 'LVEF_(%)']

dir_1 = os.listdir('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_images')
dir_2 = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_images')

dirOne = []
dirTwo = []
for d in dir_1:
    if '.DS_Store' in dir_1:
        dir_1.remove('.DS_Store')
    d = d.rstrip('_')
    dirOne.append(d)

for d in dir_2:
    if '.DS_Store' in dir_2:
        dir_2.remove('.DS_Store')
    d = d.rstrip('_')
    dirTwo.append(d)

df1 = pd.DataFrame(dirOne, columns=['index'])
df1['ID'] = df1['index'].astype(int)
df2 = pd.DataFrame(dirTwo, columns=['index'])
df2['ID'] = df2['index'].astype(int)
df_t = pd.merge(df1, df2, on=['ID'])

# Create dataframe
data = patient_df.merge(df_t, on=['ID'])
print(len(data))

def process_attributes(df):
    continuous = numerical_col_listc
    categorical = categorical_col_listc
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(df[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    testCategorical = catBinarizer.transform(df[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    testX = np.hstack([testCategorical, testContinuous])

    return (testX)

categorical_col_listm = ['Chronic_kidney_disease_(disorder)_x','Essential_hypertension_x', 'Gender_x', 'Heart_failure_(disorder)_x', 'Smoking_history_x',
'Dyslipidaemia_x', 'Myocardial_infarction_(disorder)_x', 'Diabetes_mellitus_(disorder)_x', 'Cerebrovascular_accident_(disorder)_x']
numerical_col_listm= ['Age_on_20.08.2021_x_x', 'LVEF_(%)_x']

def process_attribute(df):
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

trainx, testx = utils.patient_dataset_splitter(data, patient_key='patient_TrustNumber')
y_train = np.array(data['Event'])
y_test = np.array(df['Event_x'])
x_train = np.array(process_attributes(data))
x_test = np.array(process_attribute(df))

# fit Linear model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
print('LR Intercept:', lr_model.intercept_)
print('LR Coefficient:', lr_model.coef_)
lr_predict = lr_model.predict(x_test)
print(lr_predict[:5])
lr_preds = lr_model.predict_proba(x_test)[:,1]
print(lr_preds[:5])

print(classification_report(y_test, lr_predict))
print('Linear ROCAUC score:',roc_auc_score(y_test, lr_predict))
print('Linear Accuracy score:',accuracy_score(y_test, lr_predict))
print('Linear F1 score:',f1_score(y_test, lr_predict))
print('Linear Precision:',precision_score(y_test, lr_predict))
print('Linear Recall:',recall_score(y_test, lr_predict))

# plot confusion matrix
cm = confusion_matrix(survival_yhat, lr_predict, labels=lr_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
disp.plot()
plt.show()

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

# Test difference between models with McNemar's test
# HNN vs Image CNN
print("Evaluate HNN vs Image CNN...")
tb = mcnemar_table(y_target=y_test,
                   y_model1=pred_test_cl1,
                   y_model2=pred_test_cl2)
chi2, p = mcnemar(ary=tb, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)

# HNN vs LG
print("Evaluate HNN vs Logistic Regression...")
tb = mcnemar_table(y_target=y_test,
                   y_model1=lr_predict,
                   y_model2=pred_test_cl2)
chi2, p = mcnemar(ary=tb, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)

# Image CNN vs LG
print("Evaluate Image CNN vs Logistic Regression")
tb = mcnemar_table(y_target=y_test,
                   y_model1=pred_test_cl1,
                   y_model2=lr_predict)
chi2, p = mcnemar(ary=tb, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)

