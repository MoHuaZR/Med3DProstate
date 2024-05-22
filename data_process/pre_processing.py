from torch.utils.data import Dataset
import os
import nibabel
import numpy as np
from scipy import ndimage
import torch
import SimpleITK as sitk
import pandas as pd

class DataProcessing(object):
    def __init__(self, 
                 root_dir, 
                 phase,
                 bounding_box,
                 patient_info,
                 crop_save_path,
                 new_window_width = 350,
                 new_window_level = 35,
                 adjust_windows = True):
        self.root_dir = root_dir
        
        self.image_filenames_ct = []
        self.labels_train = []
        self.image_filenames_pet = []
        self.labels_test = []
        self.phase = phase
        
        self.bbsize = bounding_box
        self.patient_info = patient_info
        self.crop_save_path = crop_save_path
        self.new_window_width = new_window_width
        self.new_window_level = new_window_level
        self.adjust_windows = adjust_windows

        
        if self.phase == "train":
            classpath = os.path.join(self.root_dir, 'train')
        else:
            classpath = os.path.join(self.root_dir, 'test')
        classtype = os.listdir(classpath)
        
        for cls_id, cls_name in enumerate(classtype):
            class_path_ct = os.path.join(''.join([classpath, '/',cls_name]), 'CT')    
            class_path_pet = os.path.join(''.join([classpath, '/',cls_name]), 'PET')     
            for filename in os.listdir(class_path_ct):
                self.image_filenames_ct.append(os.path.join(class_path_ct, filename))
            for filename in os.listdir(class_path_pet):
                self.image_filenames_pet.append(os.path.join(class_path_pet, filename))
                self.labels_train.append(cls_id)
                self.labels_test.append(cls_id)
 
    def __len__(self):
        return len(self.image_filenames_ct)

    def cut(self):
        for idx in range(len(self.image_filenames_ct)):
            img_path_ct = self.image_filenames_ct[idx]
            img_path_pet = self.image_filenames_pet[idx]

            assert os.path.isfile(img_path_ct)
            assert os.path.isfile(img_path_pet)

            if self.phase == "train":

                self.__training_data_process__(img_path_ct, is_ct=True)
                self.__training_data_process__(img_path_pet, is_ct=False)            

                    
            elif self.phase == "test":
                # read image
                self.__testing_data_process__(img_path_ct)
                self.__testing_data_process__(img_path_pet)
        
    def __read_Nifit__(self, path):
        nii_img = nibabel.load(path)
        print("Shape of NIfTI data:", nii_img.shape)
        nii_data = nii_img.get_fdata()
        numpy_data = np.array(nii_data)
        print("Shape of NUmPy array:", numpy_data.shape)
        return numpy_data

    def __read_excel__(self, path):
        df = pd.read_excel(path)
        return df

    def __caculate_coordinate__(self, center_coordinate, bbsize):
        zstart, ystart, xstart = int(center_coordinate[2]) - bbsize[2] / 2, int(center_coordinate[1]) - bbsize[1] / 2, int(center_coordinate[0]) - bbsize[0] / 2
        ztop, ytop, xtop = int(center_coordinate[2]) + bbsize[2] / 2, int(center_coordinate[1]) + bbsize[1] / 2, int(center_coordinate[0]) + bbsize[0] / 2
        if zstart < 0:
            zstart = max(0, zstart)
            ztop = bbsize[2]
        if ystart < 0:
            ystart = max(0, ystart)
            ytop = bbsize[1]
        if xstart < 0:
            xstart = max(0, xstart)
            xtop = bbsize[0]
        return (int(zstart), int(ystart), int(xstart)), (int(ztop), int(ytop), int(xtop))

    def __save_cropped_data__(self, roi_image, name, ct_or_pet, neg_or_posi, save_path):
    
        assert save_path is not None, 'save_crop_path should not be None'
        
        print("aaaaaaaaaaa:", name)
        suffix_name = ''.join([name, '.nii.gz'])
        
        path = os.path.join(save_path, neg_or_posi, ct_or_pet, suffix_name)
        print("AAAAAAAAAAAA:", path)
      
        nibabel.save(nibabel.Nifti1Image(roi_image, affine=np.eye(4)), path)
        
        self.__get_shape__(name, path)
       
    def __get_shape__(self, patient_name , path):
        nifti_img_raw = nibabel.load(path)
        data_raw = nifti_img_raw.get_fdata()
        dimensions_raw = data_raw.shape
        print("{} shape: {}".format(patient_name, dimensions_raw))      

    def __cut_from_bb__(self, data, bbpath, name, ct_or_pet, neg_or_posi, save_path = None, bbsize = (64,64,64), mask = False):
        excel_info = self.__read_excel__(bbpath)  
        img = self.__read_Nifit__(data)
        result = excel_info[excel_info['name'] == name]
        if not result.empty:
            # 获取x、y、z的值
            x = result.iloc[0]['x']
            y = result.iloc[0]['y']
            z = result.iloc[0]['z']
        
#        print("result:", x, y, z)
        if not mask:
            patient_file = ''.join([name, '.nii.gz'])
        else:
            patient_file = ''.join([name, '_roi.nii'])
                
        coordinate = self.__caculate_coordinate__((x, y, z), bbsize)
  
        
        # z y x
        roi_image = img[coordinate[0][2]:coordinate[1][2], coordinate[0][1]:coordinate[1][1], coordinate[0][0]:coordinate[1][0]]
 
        if self.adjust_windows and ct_or_pet == 'CT':
            roi_image = self.window_transform(roi_image, self.new_window_width, self.new_window_level)
        self.__save_cropped_data__(roi_image, name, ct_or_pet, neg_or_posi, save_path)
    
    def window_transform(self, img, windowWidth, windowCenter, normal = False):
        """
        注意，这个函数的self.image一定得是float类型的，否则就无效！
        return: trucated image according to window center and window width
        """
        # print(windowWidth)
        # print(windowCenter)
        minWindow = float(windowCenter) - 0.5*float(windowWidth)
        newimg = (img - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0  
        newimg[newimg > 1] = 1
        # 将值域转到0-255之间,例如要看头颅时，我们只需将头颅的值域转换到0-255就行了
        if not normal:
            newimg = (newimg * 255).astype('uint8')
            
        return newimg
    
    def __ct_max_min_number__(self, ct_image):
        # 假设您已经加载了CT图像到一个名为 "ct_image" 的NumPy数组中
        max_value = np.max(ct_image)
        min_value = np.min(ct_image)

        print("最大值:", max_value)
        print("最小值:", min_value)
        return min_value, max_value

    def __training_data_process__(self, data, is_ct=True): 
        # crop data according net input size
        name = data.split('/')[-1].split('.')[0]
        ct_or_pet = data.split('/')[-2]
        neg_or_posi = data.split('/')[-3]
        
        if ct_or_pet == 'CT' and neg_or_posi == 'p_negative':
            patient_info_name = 'patient_names_ct_0.xlsx'
        if ct_or_pet == 'PET' and neg_or_posi == 'p_negative':
            patient_info_name = 'patient_names_pet_0.xlsx'       
        if ct_or_pet == 'CT' and neg_or_posi == 'p_positive':
            patient_info_name = 'patient_names_ct_1.xlsx'
        if ct_or_pet == 'PET' and neg_or_posi == 'p_positive':
            patient_info_name = 'patient_names_pet_1.xlsx'
            
        assert data is not None
        bbpath = os.path.join(self.patient_info, patient_info_name)
        # crop data
        path_train = os.path.join(self.crop_save_path, 'train')
        self.__cut_from_bb__(data, bbpath, name, ct_or_pet, neg_or_posi, path_train, self.bbsize) 
        
    def __testing_data_process__(self, data, is_ct=False): 
        # crop data according net input size
        name = data.split('/')[-1].split('.')[0]
        ct_or_pet = data.split('/')[-2]
        neg_or_posi = data.split('/')[-3]
   
        if ct_or_pet == 'CT' and neg_or_posi == 'p_negative':
            patient_info_name = 'patient_names_ct_0.xlsx'
        if ct_or_pet == 'PET' and neg_or_posi == 'p_negative':
            patient_info_name = 'patient_names_pet_0.xlsx'       
        if ct_or_pet == 'CT' and neg_or_posi == 'p_positive':
            patient_info_name = 'patient_names_ct_1.xlsx'
        if ct_or_pet == 'PET' and neg_or_posi == 'p_positive':
            patient_info_name = 'patient_names_pet_1.xlsx'
            
        assert data is not None
        bbpath = os.path.join(self.patient_info, patient_info_name)
        # crop data
        path_test = os.path.join(self.crop_save_path, 'test')
        self.__cut_from_bb__(data, bbpath, name, ct_or_pet, neg_or_posi, path_test, self.bbsize) 

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
if __name__ =='__main__':
    root_dir = '/home/cavin/Experiment/ZR/data/Prostate_100_50_all_0_psma'
    phase = 'test'
    bounding_box = (64,64,64)
    patient_info = '/home/cavin/Experiment/ZR/MedicalNet-master/patient_info/bounding_box_info'
    crop_save_path = '/home/cavin/Experiment/ZR/MedicalNet-master/cropped_data/cropped_normal_64_64_50test_new_v2'
    new_window_width = 325
    new_window_level = 45          
    adjust_windows = True
    prostate = DataProcessing(root_dir,  
                              phase, 
                              bounding_box, 
                              patient_info, 
                              crop_save_path, 
                              new_window_width, 
                              new_window_level,
                              adjust_windows)
    prostate.cut()