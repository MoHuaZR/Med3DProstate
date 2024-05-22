'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default='/home/cavin/Experiment/ZR/Data_Process/crop_64_prostate_ch_v2',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--excel_path',
        default='/home/cavin/Experiment/ZR/MedicalNet-master/patient_info/excel_info/patients_info.xlsx',
        type=str,
        help='patient_info_excel')    
    parser.add_argument(
        '--json_path',
        default='/home/cavin/Experiment/ZR/MedicalNet-master/patient_info/json_info/patient_english.json',
        type=str,
        help='json_path')    
    parser.add_argument(
        '--mixed_clinic',
        action='store_true',
        help='mix the clnical data')  
    parser.add_argument(
        '--focalloss',
        action='store_true',
        help='focal loss') 
    parser.add_argument(
        '--regualarization',
        action='store_true',
        help='loss regualarization') 
    parser.add_argument(
        '--weight_for_negative_class',
        default = 2.0,
        type=float,
        help='weight of negative') 
    parser.add_argument(
        '--clinic_dimension',
        default=6,
        type=int,
        help='the dimension of clnical data')  
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--if_transform',
        default=False,
        type=bool,
        help="transform or not"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.0001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=16,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=200, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--n_epochs',
        default=1000,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--validation_split',
        default=0.3,
        type=float,
        help='percentage of validation data')
    parser.add_argument(
        '--input_D',
    default=64,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=64,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=64,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        # default='/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/old_pth/depth_101_v2/resnet_50_171_epoch_636_batch_11_auc_0.8814102564102564.pth.tar',
        # default='/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v6/resnet_50_64_epoch_140_batch_8_auc_0.6875.pth.tar',
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v7/resnet_50_39_epoch_100_batch_11_auc_0.8080357142857143.pth.tar',
        # default='/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_29_epoch_59_batch_17_auc_0.8348214285714286.pth.tar',
        
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_64_epoch_140_batch_17_auc_0.6651785714285714.pth.tar',
        #76.6667
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_61_epoch_130_batch_17_auc_0.7410714285714286.pth.tar',
        #75.8333
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_70_epoch_151_batch_17_auc_0.7321428571428571.pth.tar', 
        #75.8333
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_73_epoch_155_batch_17_auc_0.7991071428571428.pth.tar',
        #75.8333
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_78_epoch_164_batch_17_auc_0.7321428571428571.pth.tar',
        #75.8333
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_85_epoch_180_batch_17_auc_0.7723214285714287.pth.tar',
        #73.3333
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_86_epoch_184_batch_17_auc_0.6785714285714286.pth.tar',
        #77.4999
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_92_epoch_192_batch_17_auc_0.7098214285714286.pth.tar',
        #74.6667
        # default = '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_95_epoch_201_batch_17_auc_0.7633928571428572.pth.tar',
        #74.1667
        # default= '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v8/resnet_50_103_epoch_218_batch_17_auc_0.7767857142857143.pth.tar',
        #78.3333
        
        # default= '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v11/resnet_50_82_epoch_170_batch_1_auc_0.9027777777777777.pth.tar',
        # 0.75
        # default='/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v11/resnet_50_101_epoch_213_batch_1_auc_0.9444444444444444.pth.tar',
        # 0.75
        # default='/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v11/resnet_50_103_epoch_217_batch_1_auc_0.9444444444444444.pth.tar',
        # 0.73
        # default= '/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/new_pth/depth_50_v11/resnet_50_72_epoch_143_batch_1_auc_0.9027777777777777.pth.tar',
        # 0.725
        default= '',
        type=str,
        help= 'Path for resume model.'
    )
    
    parser.add_argument(
        '--pretrain_path',
        # default='/home/cavin/Experiment/ZR/MedicalNet-master/pretrain/resnet_50.pth',
        default='',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        #default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        #default=['conv_seg'],
        default= ['mlp'],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(  #3407
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    args.save_folder = "/home/cavin/Experiment/ZR/MedicalNet-master/trails_classification/models/depth_50_64_v1/{}_{}".format(args.model, args.model_depth)
    
    return args
