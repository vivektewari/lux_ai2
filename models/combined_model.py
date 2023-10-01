import torch

from baseline_models import *


class model_extension3(fc_model):
    def add_premodel(self,model):
        pass

    def forward(self, x):
        x = super().forward(x)
        return x  # torch.sigmoid(x)
class model_extension2(ResNet):
    def add_premodel(self,model):pass
    def forward(self, x):

        x = super().forward(x)
        return x  # torch.sigmoid(x)
class model_extension(FeatureExtractor_baseline):
    def add_premodel(self,model):
        self.premodel=model
        #self.batch_normalizer = nn.BatchNorm1d(affine=False, num_features=53)
        self.count=0
        # self.rol_mean=torch.zeros((1,53))
        # self.rol_std=torch.zeros((1,53))
        # self.mean_=[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #  0.0000e+00, 0.0000e+00, 0.0000e+00, 7.1205e-03, 1.4403e-02, 2.9991e-03,
        #  6.0586e-03, 6.9343e-03, 1.2682e-02, 2.1726e-04, 9.9901e-04, 2.4627e-02,
        #  0.0000e+00, 5.4544e-01, 5.4544e-01, 2.3167e-01, 0.0000e+00, 8.3723e-02,
        #  8.3723e-02, 6.0000e-02, 6.0000e-02, 0.0000e+00, 0.0000e+00, 2.1523e-02,
        #  3.3528e-01, 1.2024e-02, 1.3226e-01, 1.0674e-02, 1.7261e-02, 2.1393e-04,
        #  3.4607e-04, 0.0000e+00, 3.6537e-04, 3.6899e-03, 6.8265e-03, 0.0000e+00,
        #  0.0000e+00, 6.5447e-03, 1.1153e-02, 2.2793e-03, 3.5371e-03]
        self.std_=1

    def forward(self,x):
        #x = torch.flatten(x, start_dim=-2)


        # self.count+=1
        # self.rol_mean+=torch.mean(x,dim=(2,3))
        # self.rol_std += torch.std(x, dim=(2, 3))
        #x=(x-self.rol_mean.reshape(1,53,1,1))/(self.rol_mean.reshape(1,53,1,1)+0.00001) #/(self.rol_std.reshape(1,53,1,1)+0.01)
        # if (self.count-1)%1000==456:
        #     #print(self.rol_mean/self.count)
        #     print(torch.std(torch.std(x, dim=(2, 3))))
        #print(x.shape)
        # x = self.batch_normalizer(x)
        # print(torch.std(torch.std(x,dim=2)))
        # unflatten = torch.nn.Unflatten(2, (12, 12))
        # x = unflatten(x)
        x[0,:15,:,:]=0
        x[0, 17:, :, :] = 0
        x=self.premodel(x)
        x=super().forward(x)
        return x# torch.sigmoid(x)

    def __init2__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10],premodel=None):
        super().__init__()
        self.premodel = premodel
        self.num_blocks = len(channels)
        self.start_channel = start_channel
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = tuple(input_image_dim)
        self.fc1_p = fc1_p
        self.mode_train = 1
        self.activation_l = torch.nn.LeakyReLU()#ReLU()
        self.activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0)

        last_channel = start_channel
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim2()
        if self.fc1_p[0] is not None:
            self.fc1 = nn.Linear(conv_dim[0], fc1_p[0], bias=True)
            self.fc2 = nn.Linear(fc1_p[0], fc1_p[1], bias=True)
        else:
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=fc1_p[1],
                                              kernel_size=(1, 1), stride=(1, 1),
                                              pool_size=(conv_dim[1][-2], conv_dim[1][-1]), padding=0))
            self.num_blocks += 1

        self.init_weight()
        count_parameters(self)
    def get_conv_output_dim2(self):
        input_ = torch.Tensor(np.zeros((1, self.start_channel) + self.input_image_dim))
        x=self.premodel(input_)
        x = self.cnn_feature_extractor(x)
        #print(x.shape)
        return len(x.flatten()), x.shape


