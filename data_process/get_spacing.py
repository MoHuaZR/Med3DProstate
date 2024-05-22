import nibabel as nib
import numpy as np
import os


nii_path = '/home/cavin/Experiment/ZR/MedicalNet-master/cropped_data/cropped_normal_64_64_50test/train/p_negative/CT'
child_pat_list = os.listdir(nii_path)
# 读取NIfTI文件

path_list = [os.path.join(nii_path, path) for path in child_pat_list]

print(path_list)
for item in path_list:
        
    img = nib.load(item)

    header = img.header

    spacing = header['pixdim'][1:4]
    print("CT Image Spacing:", spacing)