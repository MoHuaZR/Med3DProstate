import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
class ClinicalModel(nn.Module):
    def __init__(self, clinic_dimension, out, drop=0., indenty = False):
        super(ClinicalModel, self).__init__()
        self.clinic_dimension = clinic_dimension
        self.out = out
        self.fc1 = nn.Linear(self.clinic_dimension, 768)
        self.fc2 = nn.Linear(768, self.out)
        self.drop = nn.Dropout(drop)
        self.indenty = indenty
        
    def forward(self, x):
        if self.indenty:
            return x
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        
        return x

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda = False,
                 mix_clinic = False,
                 clinic_dimension = 7):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.mix_clinic = mix_clinic
        self.clinic_dimension = clinic_dimension
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class FeatureFusionModel(nn.Module):
    def __init__(self, 
                 block,
                 layers,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False,
                 mix_clinic = False,
                 clinic_dimension = 7):
        super(FeatureFusionModel, self).__init__()
        self.out = 2048
        self.num_seg_classes = num_seg_classes
        self.shortcut_type = shortcut_type
        self.no_cuda = no_cuda
        self.mix_clinic = mix_clinic
        self.clinic_dimension = clinic_dimension
        self.block = block
        self.layers = layers
        self.drop = 0.
        self.indenty = True
        
        self.clinical_model = ClinicalModel(clinic_dimension, self.out, self.drop, self.indenty)
        self.resnet1 = ResNet(
                            self.block,
                            self.layers,
                            self.shortcut_type,
                            self.no_cuda,
                            self.mix_clinic,
                            self.clinic_dimension
        )
        self.resnet2 = ResNet(
                            self.block,
                            self.layers,
                            self.shortcut_type,
                            self.no_cuda,
                            self.mix_clinic,
                            self.clinic_dimension
        )
        
        self.avg_pool_classfication  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten_classfication = nn.Flatten(start_dim=1)
        if self.mix_clinic:
            if self.indenty:
                self.out = self.clinic_dimension
            self.mlp = Mlp(in_features=2* 512 * block.expansion + self.out, hidden_features=768, out_features=num_seg_classes)
            self.fc_classfication = nn.Linear(in_features=2* 512 * block.expansion + self.clinic_dimension, out_features=num_seg_classes, bias=True)
        else:
            self.mlp = Mlp(in_features=2* 512 * block.expansion , hidden_features=768, out_features=num_seg_classes)
            self.fc_classfication = nn.Linear(in_features=2* 512 * block.expansion, out_features=num_seg_classes, bias=True)
        
        self.mlp1 = Mlp(2048, 768, 2)
    def forward(self, x, y, tensor_clinic):
        ct = self.flatten_classfication(self.avg_pool_classfication(self.resnet1(x)))
        # pet = self.flatten_classfication(self.avg_pool_classfication(self.resnet2(y)))
        ct = self.mlp1(ct)
        # pet = self.mlp1(pet)
        # print(pet.shape)
        
        return ct
        # clda = self.clinical_model(tensor_clinic)
        
        # return ct
        
        
        if self.mix_clinic:
            x = torch.cat((clda, ct, pet), dim=1)
        else:
            x = torch.cat((clda, ct), dim=1)
            
        # print("aaaaaaaa:", x.shape)
        
        
        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = FeatureFusionModel(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = FeatureFusionModel(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = FeatureFusionModel(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = FeatureFusionModel(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = FeatureFusionModel(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = FeatureFusionModel(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = FeatureFusionModel(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
