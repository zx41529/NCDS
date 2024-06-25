import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
import torch.nn.functional as F
 
from torch.hub import load_state_dict_from_url
from senet import SEBasicBlock, SEBottleneck
from dynamic_queue import dynamic_queue


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2','se_resnet18',
           'se_resnet34','se_resnet50','se_resnet101','se_resnet152']

 
projection_dim =1000
num_projection_MLP = 3
print("wide_resnet_num_projection_MLP:",num_projection_MLP)



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'se_resnet50' : "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl",
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = num_projection_MLP
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class SepConv(nn.Module):   

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)




class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')      
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)   
        self.bn1 = norm_layer(planes)     
        self.relu = nn.ReLU(inplace=True)  
        self.conv2 = conv3x3(planes, planes)  
        self.bn2 = norm_layer(planes)
        self.downsample = downsample   
        self.stride = stride  

 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  
 
        out = self.conv2(out)
        out = self.bn2(out) 
 
        if self.downsample is not None:    
            identity = self.downsample(x)   
 
        out += identity             
        out = self.relu(out)
 
        return out

 
class Bottleneck(nn.Module):
    expansion = 4  
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  
        
        self.conv1 = conv1x1(inplanes, width)  
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x   
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)   
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) 
 
        out = self.conv3(out)
        out = self.bn3(out)  
 
       
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  
 
        self.inplanes = 64  
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
       
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) 

        self.bn1 = norm_layer(self.inplanes)   
        self.relu = nn.ReLU(inplace=True)  
      
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
      
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
           
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.auxiliary3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
         
            nn.AdaptiveAvgPool2d((1, 1))

            
        )
 


        
        self.auxiliary4 = nn.AdaptiveAvgPool2d((1, 1))
       
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc1 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128)
            ) 
        self.fc2 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128)
            ) 
        self.fc3 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128)
            ) 
        self.fc4 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128)
            ) 

        self.fc1_projector = projection_MLP(512 * block.expansion)
        self.fc2_projector = projection_MLP(512 * block.expansion)
        self.fc3_projector = projection_MLP(512 * block.expansion)
        self.fc4_projector = projection_MLP(512 * block.expansion)

        self.fc1_predictor = prediction_MLP()
        self.fc2_predictor = prediction_MLP()
        self.fc3_predictor = prediction_MLP()
        self.fc4_predictor = prediction_MLP()

          

        self.queues = dynamic_queue()


      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):     
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation  
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:  
            downsample = nn.Sequential(    
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
 
        layers = []                  
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))  
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):        
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        feature_list.append(x)
        x = self.layer2(x)
        feature_list.append(x)
        x = self.layer3(x)
        feature_list.append(x)
        x = self.layer4(x)
        feature_list.append(x)

        out1_feature = self.auxiliary1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.auxiliary4(feature_list[3]).view(x.size(0), -1)


        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        out = self.fc(out4_feature)

        out1_z = self.fc1_projector(out1_feature)  
        out2_z = self.fc2_projector(out2_feature)  
        out3_z = self.fc3_projector(out3_feature)  
        out4_z = self.fc4_projector(out4_feature)  
        
        out1_h = self.fc1_predictor(out1_z)  
        out2_h = self.fc2_predictor(out2_z)  
        out3_h = self.fc3_predictor(out3_z)  
        out4_h = self.fc4_predictor(out4_z)  

        z_list = [out4_z, out3_z, out2_z, out1_z] 
        h_list = [out4_h, out3_h, out2_h, out1_h] 
        
        feat_list=[out4, out3, out2, out1]
        for index in range(len(feat_list)):                                 
            feat_list[index] = F.normalize(feat_list[index], dim=1)
       
        qs = [self.queues.queue4.clone(),self.queues.queue3.clone(),self.queues.queue2.clone(),self.queues.queue1.clone(),self.queues.label.clone().contiguous().view(-1, 1) ]     
        
        
        if self.training:
            self.queues.dequeue_and_enqueue(feat_list[3][labels.shape[0]:,:],feat_list[2][labels.shape[0]:,:],feat_list[1][labels.shape[0]:,:],feat_list[0][labels.shape[0]:,:],labels)
            return out, z_list, h_list, feat_list, qs
        else:
            return out
 
 
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
 
 
def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
 
 
def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
 
 
def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
 
 
def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
 
 
def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
 
 
def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
 
 
def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
 
 
def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
 
 
def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)





def se_resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    return _resnet('se_resnet18', SEBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def se_resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    return _resnet('se_resnet34', SEBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def se_resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    # if pretrained:
    #     model.load_state_dict(load_state_dict_from_url(
    #         "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return _resnet('se_resnet50', SEBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def se_resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    return _resnet('se_resnet101', SEBottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def se_resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    return _resnet('se_resnet152', SEBottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)




# print(wide_resnet50_2)


