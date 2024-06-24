import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
import torch.nn.functional as F
 
from torch.hub import load_state_dict_from_url
from senet import SEBasicBlock, SEBottleneck
from dynamic_queue import dynamic_queue


#其中torch.nn 为其提供基础函数，model_zoo提供权重数据的下载。
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2','se_resnet18',
           'se_resnet34','se_resnet50','se_resnet101','se_resnet152']
#ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。
 
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
#groups: 控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
# dilation=1（也就是 padding）          groups 是分组卷积参数，这里 groups=1 相当于没有分组   第一个3*3的主要作用是在以后高维中做卷积提取信息，第二个1*1的作用主要是进行升降维的。
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
#注意：这里bias设置为False,原因是：下面使用了Batch Normalization，而其对隐藏层  有去均值的操作，所以这里的常数项 可以消去
# 因为Batch Normalization有一个操作，所以上面的数值效果是能由所替代的,因此我们在使用Batch Norm的时候，可以忽略各隐藏层的常数项  。这样在使用梯度下降算法时，只用对  ， 和  进行迭代更新



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


class SepConv(nn.Module):   #将长高各减少一半，并增加通道数

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2  深度和逐点卷积，下采样2
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



#BasicBlock是为resnet18、34设计的，由于较浅层的结构可以不使用Bottleneck。
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   # #BatchNorm2d最常用于卷积网络中(防止梯度消失或爆炸)，设置的参数就是卷积的输出通道数
            #计算各个维度的标准和方差，进行归一化操作
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')       #为什么要设置这些限制
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)   #卷积操作，输入通道，输出通道，步长
        self.bn1 = norm_layer(planes)     #防止梯度爆炸或消失，planes就是卷积一次之后的输出通道数？为什么要对输出的通道数进行防爆呢
        self.relu = nn.ReLU(inplace=True)  #inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出。
        self.conv2 = conv3x3(planes, planes)  #conv层的时候通道数是一样的都是64的倍数，但是下一层的时候会改变，所以这里用了inplaces和planes两个变量
        self.bn2 = norm_layer(planes)
        self.downsample = downsample   #下采样
        self.stride = stride   #步长
#解读：这个结构就是由两个3*3的结构为主加上bn和一次relu激活组成。其中有个downsample是由于有x+out的操作，要保证这两个可以加起来所以对原始输入的x进行downsample。
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  #一次卷积，防爆，激活
 
        out = self.conv2(out)
        out = self.bn2(out)  #第二次卷积，防爆
 
        if self.downsample is not None:     #当连接的维度不同时，使用1*1的卷积核将低维转成高维，然后才能进行相加
            identity = self.downsample(x)   #就是在进行下采样，如果需要的话
 
        out += identity               #这个时候就会用到残差网络的特点，f(x)+x # 实现H(x)=F(x)+x或H(x)=F(x)+Wx
        out = self.relu(out)
 
        return out
#看到代码 self.downsample = downsample，在默认情况downsample=None，表示不做downsample，但有一个情况需要做，就是一个 BasicBlock的分支x要与output相加时，若x和output的通道数不一样，则要做一个downsample，
# 剧透一下，在resnet里的downsample就是用一个1x1的卷积核处理，变成想要的通道数。为什么要这样做？因为最后要x要和output相加啊， 通道不同相加不了。所以downsample是专门用来改变x的通道数的。
 
class Bottleneck(nn.Module):
    #expansion 是对输出通道数的倍乘，注意在基础版本 BasicBlock 中 expansion 是 1，此时相当于没有倍乘，输出的通道数就等于 planes。
    expansion = 4  #一层里面最终输出时四倍膨胀
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  #这个值应该是变化的
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  #with在这里应该是改变输入的维度
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  #输入输出的通道一样
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x   #shotcut
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)   #1x1卷积
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)  #3x3卷积
 
        out = self.conv3(out)
        out = self.bn3(out)  #1x1 归一
 
        #不管是BasicBlock还是Bottleneck，最后都会做一个判断是否需要给x做downsample，因为必须要把x的通道数变成与主枝的输出的通道一致，才能相加。
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
        self._norm_layer = norm_layer  #为什么这么做，是因为在make函数中也要用到norm_layer，所以将这个放到了self中
 
        self.inplanes = 64  #设置默认输入通道
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
        # 原始resnet
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) #7x7  输入3  输出inplanes  步长为2  填充为3   偏移量为false
        # cifa适配版 减少cifar数据集上ResNet的内核大小和步长
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)

        self.bn1 = norm_layer(self.inplanes)   #归一化防爆
        self.relu = nn.ReLU(inplace=True)  #激活函数替换
        # cifa适配版 .删除cifar数据集上ResNet的maxpooling层。
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #最大池化3x3 步长为2 填充为1
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
            # cifa适配版
            # nn.AvgPool2d(4, 4)
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
            # cifa适配版
            # nn.AvgPool2d(4, 4)
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.auxiliary3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # cifa适配版
            # nn.AvgPool2d(4, 4)
            nn.AdaptiveAvgPool2d((1, 1))

            
        )
        # cifa适配版
        # self.auxiliary4 = nn.AvgPool2d(4, 4)


        
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

        #  #有监督mlp
        # self.fc4_out = nn.Sequential(
        #     nn.Linear(512 * block.expansion, 256 * block.expansion, bias=False),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256 * block.expansion, num_classes, bias=False),
        # )                         #改动 对layer4 接全连接   

        self.queues = dynamic_queue()


        # 对卷积和与BN层初始化，论文中也提到过
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
 
#_make_layer 方法的第一个输入参数 block 选择要使用的模块是 BasicBlock 还是 Bottleneck 类，第二个输入参数 planes 是该模块的输出通道数，第三个输入参数 blocks 是每个 blocks 中包含多少个 residual 子结构。
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):      #planes参数是“基准通道数”，不是输出通道数！！！不是输出通道数！！！不是输出通道数！！！)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation  #填充？
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:  #如果stride不等于1或者维度不匹配的时候的downsample，可以看到也是用过一个1*1的操作来进行升维的，然后对其进行一次BN操作
            downsample = nn.Sequential(    #一个时序器
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
 
        layers = []                    #[3，4，6，3]表示按次序生成3个Bottleneck，4个Bottleneck，6个Bottleneck，3个Bottleneck。
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))  #该部分是将每个blocks的第一个residual结构保存在layers列表中
        #这里分两个block是因为要将一整个Lyaer进行output size那里，维度是依次下降两倍的，第一个是设置了stride=2所以维度下降一半，剩下的不需要进行维度下降，都是一样的维度
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):        #该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)
#ResNet 共有五个阶段，其中第一阶段为一个 7*7 的卷积，stride = 2，padding = 3，然后经过 BN、ReLU 和 maxpooling，此时特征图的尺寸已成为输入的 1/4
#接下来是四个阶段，也就是代码中 layer1，layer2，layer3，layer4。这里用 _make_layer 函数产生四个 Layer，需要用户输入每个 layer 的 block 数目（ 即layers列表 )以及采用的 block 类型（基础版 BasicBlock 还是 Bottleneck 版）
    def forward(self, x, labels=None):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)   #第一阶段进行普通卷积 变成原来1/4
 
        #其实所谓的layer1，2，3，4都是由不同参数的_make_layer()方法得到的。看_make_layer()的参数，发现了layers[0~3]就是上面输入的[3，4，6，3]，即layers[0]是3，layers[1]是4，layers[2]是6，layers[3]是3。
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


        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # out = self.fc4_out(out4_feature)
        out = self.fc(out4_feature)

        
        #mlp 投影层
        out1_z = self.fc1_projector(out1_feature)  
        out2_z = self.fc2_projector(out2_feature)  
        out3_z = self.fc3_projector(out3_feature)  
        out4_z = self.fc4_projector(out4_feature)  
        
        #mlp 预测层
        out1_h = self.fc1_predictor(out1_z)  
        out2_h = self.fc2_predictor(out2_z)  
        out3_h = self.fc3_predictor(out3_z)  
        out4_h = self.fc4_predictor(out4_z)  

        z_list = [out4_z, out3_z, out2_z, out1_z] 
        h_list = [out4_h, out3_h, out2_h, out1_h] 
        # feat_list=[out4_feature, out3_feature, out2_feature, out1_feature]
        feat_list=[out4, out3, out2, out1]
        for index in range(len(feat_list)):                                 
            feat_list[index] = F.normalize(feat_list[index], dim=1)
        # for index in range(len(feat_list)):                                 
        #     feat_list[index] = F.normalize(feat_list[index], dim=1)
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


