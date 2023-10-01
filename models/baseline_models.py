
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from prettytable import PrettyTable

# from multibox_loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(6, 6),
                 stride=(1, 1), padding=(5, 5), pool_size=(2, 2)):
        super().__init__()
        self.pool_size = pool_size
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(np.array(kernel_size) + np.array([0, 0])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 0])),
            bias=True)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input1, pool_size=None, pool_type='max'):
        if pool_size is None: pool_size = self.pool_size
        x = input1

        x = self.conv(input1)  # F.relu_(self.conv(input1))
        #
        #x = self.bn2(self.conv2(x))
        # if pool_type == 'max':
        #     x = F.max_pool2(x, kernel_size=pool_size)
        return x


class FeatureExtractor_baseline(nn.Module):
    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        super().__init__()
        self.num_blocks = len(channels)
        self.start_channel = start_channel
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = tuple(input_image_dim)
        self.fc1_p = fc1_p
        self.mode_train = 1
        self.activation_l = torch.nn.ReLU() #torch.nn.LeakyReLU()
        self.activation = torch.nn.Tanh() #torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0)

        last_channel = start_channel
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim()
        if self.fc1_p[0] is not None:
            self.fc1 = nn.Linear(conv_dim[0], fc1_p[0], bias=True)
            self.fc2 = nn.Linear(fc1_p[0], fc1_p[1], bias=True)
            if  False and self.fc1_p[1] is not None:
                self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=fc1_p[1],
                                                  kernel_size=(1, 1), stride=(1, 1),
                                                  pool_size=(conv_dim[1][-2], conv_dim[1][-1]), padding=0))
                self.num_blocks += 1

        self.init_weight()
        count_parameters(self)

    def get_conv_output_dim(self):
        input_ = torch.Tensor(np.zeros((1, self.start_channel) + self.input_image_dim))
        x = self.cnn_feature_extractor(input_)
        print(x.shape)
        return len(x.flatten()), x.shape

    @staticmethod
    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight )
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_blocks):
            self.init_layer(self.conv_blocks[i].conv)

        if self.fc1_p[0] is not None:
            self.init_layer(self.fc1)
            self.init_layer(self.fc2)
        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)

    def cnn_feature_extractor(self, x):
        # input 501*64
        for i in range(self.num_blocks):
            if torch.isnan(self.conv_blocks[i](x)).any():
                j=0

            x = self.conv_blocks[i](x)


            # x_70=torch.quantile(x, 0.7)
            # x_50 = torch.quantile(x, 1)
            # x=self.activation_l((x-x_50-1)/max(x_50,0.01))
            if i < (len(self.conv_blocks) - 1):
                x = self.activation_l(x)
                #print(torch.std(x),torch.min(x),torch.max(x))
                # if torch.std(x)>0.0001:#fix: bad fix as valueue wer becoming nan
                #      x = (x - torch.mean(x)) / torch.std(x)


            if self.mode_train == 1:
                x = self.dropout(x)
            # x=torch.clamp(x,100,-100)

        return x

    def forward(self, input_):
        #print(input_ [0,36,0,0])

        x = input_ * 1.0

        #x =torch.unflatten(x,2,(12,12))
        #temp=torch.zeros(1,53,12,12)
        #temp[0,36,0,0]=1
        #x=x*temp
        #todo remove below 2 lines
        # x[0, 0:29, 0, 0] = x[0, 0:29, 0, 0] * 0.0
        # x[0, 31:, 0, 0] = x[0, 31:, 0, 0] * 0.0

        #x = input_/torch.max(input_)
        x = self.cnn_feature_extractor(x)
        #x=(x-torch.mean(x))/torch.std(x)
        #print(1, torch.max(x))
        #x=x+input_[:,:,:9,:9]#F.pad(x, (0,3,3,0,3,0), "constant", 0)F.pad(x, (3,0,3,0))
        #print(2, torch.max(x),torch.max(input_[:,:,:9,:9]))
        if self.fc1_p[0] is None and self.fc1_p[1] is None:
            #x = x.flatten(start_dim=1, end_dim=-1)
            return self.activation_l(x)
        else:x = x.flatten(start_dim=1, end_dim=-1)


        if self.fc1_p[0] is not None:

            x = self.activation_l(x)
            x = self.activation_l(self.fc1(x))
            #print(3, torch.max(x))
            # if max(x.shape)>1:
            #      x = (x - torch.mean(x)) /torch.torch.std(x)
            if self.mode_train == 1:
                x = self.dropout(x)
            #x=torch.abs(x)*0+1#todo remove
        if self.fc1_p[1] is not None:x = self.fc2(x)
        else:x = x
       # x=self.activation(x)
        return torch.clip(x,-100,100).flatten()# F.tanh(x.flatten())


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=2,stride=1,padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,blocks, channels,start_channel, input_image_dim,convs,pads,strides,pools,fc1_p):
        super(ResNet, self).__init__()
        self.num_blocks=len(blocks)
        self.inplanes = 128
        self.blocks = nn.ModuleList()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=start_channel,out_channels= 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        last_channel=128
        for i in range(self.num_blocks):
            self.blocks.append(self._make_layer(block=ResidualBlock,blocks=blocks[i],in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=convs[i], stride=strides[i],
                                              padding=pads[i]))
            last_channel=channels[i]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=12, stride=1)
        self.fc = nn.Linear(fc1_p[0], fc1_p[1])
        count_parameters(self)

    def _make_layer(self,block,in_channels,out_channels, blocks,  kernel_size,padding=1, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = nn.ModuleList()
        for i in range(1,blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride,padding=padding))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) #input_dim*start_channels->input_dim*128channels
        x = self.maxpool(x)#input_dim*128channels->input_dim*128channels
        for i in range(self.num_blocks):
            x = self.blocks[i](x) #input_dim*128channels->input_dim*channels[i]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class fc_model(nn.Module):
    def __init__(self,input_nodes,output_nodes):
        super().__init__()
        self.fc1 = nn.Linear(input_nodes, output_nodes, bias=True)
        #self.fc1.bias.data.fill_(0.0)
        #self.fc1.weight.data.fill_(0.1)
        count_parameters(self)
    def forward(self,x):
        return self.fc1(x[0,15:17,:,:].flatten())
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params