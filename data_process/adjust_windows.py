import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def adjust_window(img_data, window_width, window_level):
    """
    调整图像的窗宽窗位。
    
    Parameters:
    - img_data: CT图像数据的Numpy数组
    - window_width: 新的窗宽
    - window_level: 新的窗位
    
    Returns:
    - 调整后的CT图像数据
    """
    min_value = window_level - window_width / 2.0
    max_value = window_level + window_width / 2.0
    adjusted_img = np.clip(img_data, min_value, max_value)
    return adjusted_img

def main():
    # 读取nii.gz文件
    ct_image_path = '/home/cavin/Experiment/ZR/data/Prostate_100_43/train/p_negative/CT/caichangyuan.nii.gz'
    ct_img = nib.load(ct_image_path)
    ct_data = ct_img.get_fdata()

    # 设置新的窗宽窗位值
    new_window_width = 350
    new_window_level = 35

    # 调整窗宽窗位
    adjusted_ct_data = adjust_window(ct_data, new_window_width, new_window_level)

    # 保存调整后的图像
    adjusted_ct_img = nib.Nifti1Image(adjusted_ct_data, ct_img.affine)
    nib.save(adjusted_ct_img, 'adjusted_ct_image.nii.gz')

    # 显示原始和调整后的图像（示例，您可以根据需要自定义）
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ct_data[:, :, ct_data.shape[2] // 2], cmap='gray', vmin=0, vmax=255)
    plt.title('Original CT Image')

    plt.subplot(1, 2, 2)
    plt.imshow(adjusted_ct_data[:, :, adjusted_ct_data.shape[2] // 2], cmap='gray', vmin=0, vmax=255)
    plt.title('Adjusted CT Image (Window Width={}, Window Level={})'.format(new_window_width, new_window_level))

    plt.show()

if __name__ == "__main__":
    main()