import os
import shutil

def search_and_copy_files(source_folder, target_copy_source_path, target_to_folder_path):
    # 获取源文件夹中的所有文件
    source_files = os.listdir(source_folder)

    for file_name in source_files:
        # 构建源文件的完整路径
        target_copy_source_path_all = os.path.join(target_copy_source_path, file_name)

        # 构建目标文件夹中的文件路径
        target_file_path = os.path.join(target_to_folder_path, file_name)

        # 判断源文件是否存在于目标文件夹中
        if os.path.exists(target_copy_source_path_all):
            # 复制文件到其他文件夹
            shutil.copy(target_copy_source_path_all, target_file_path)
            print(f"文件 {file_name} 已复制到目标文件夹")
        else:
            print(f"文件 {file_name} 不存在于目标文件夹")

# 指定源文件夹和目标文件夹的路径
source_folder_path = "/home/cavin/Experiment/ZR/MedicalNet-master/cropped_data/test/p_positive/CT"
target_copy_source_path = '/home/cavin/Experiment/ZR/MedicalNet-master-OLD/cropped_data_64/train/p_positive/CT'
target_to_folder_path = "/home/cavin/Experiment/ZR/MedicalNet-master/cropped_data_64/test/p_positive/CT"

# 调用函数进行检索和复制
search_and_copy_files(source_folder_path, target_copy_source_path, target_to_folder_path)

