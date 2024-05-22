import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from itertools import cycle
import json

# 读取Excel表格
excel_file_path = '/home/cavin/Experiment/ZR/MedicalNet-master/compare_roc/patient_info_new_sorted.xlsx'
excel_data = pd.read_excel(excel_file_path)

# 读取文件夹中的文件列表
folder_path = '/home/cavin/Experiment/ZR/MedicalNet-master/compare_roc/patient_info/'
file_list = os.listdir(folder_path)

# 初始化存储数据的列表
psma_values = []
mskcc_values = []
labels = []

json_path = 'patient_info/json_info/patient_new_enlish_V2.json'

with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历文件列表，从Excel中提取相应的属性值
for file_name in file_list:
    patient_id = os.path.splitext(file_name)[0]
    name = patient_id.split('.')[0]
    name_to_find = data['test'][name]
    patient_data = excel_data[excel_data['姓名'] == name_to_find]

    if not patient_data.empty:
        psma_values.append(patient_data['PSMA'].values[0])
        mskcc_values.append(patient_data['MSKCC'].values[0])
        labels.append(patient_data['淋巴结转移情况'].values[0])

# 将数据转换为NumPy数组
psma_values = np.array(psma_values)
mskcc_values = np.array(mskcc_values)
labels = np.array(labels)

# 将标签二值化
labels = label_binarize(labels, classes=[0, 1])

# # 拆分数据集为训练集和测试集
# psma_train, psma_test, mskcc_train, mskcc_test, labels_train, labels_test = train_test_split(
#     psma_values, mskcc_values, labels, test_size=0.8, random_state=42
# )

# 计算PSMA和MSKCC的ROC曲线
fpr_psma, tpr_psma, _ = roc_curve(labels, psma_values)
roc_auc_psma = auc(fpr_psma, tpr_psma)

fpr_mskcc, tpr_mskcc, _ = roc_curve(labels, mskcc_values)
roc_auc_mskcc = auc(fpr_mskcc, tpr_mskcc)

# 绘制ROC曲线
plt.figure(figsize=(8, 8))

plt.plot(fpr_psma, tpr_psma, color='blue', lw=2, label='ROC curve PSMA (area = {:.2f})'.format(roc_auc_psma))
plt.plot(fpr_mskcc, tpr_mskcc, color='red', lw=2, label='ROC curve MSKCC (area = {:.2f})'.format(roc_auc_mskcc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for PSMA and MSKCC')
plt.legend(loc='lower right')
plt.show()