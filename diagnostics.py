import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
pre_trained_model='/home/pooja/PycharmProjects/lux_ai/outputs/monobeast_output/rl_model_200.pth'
checkpoint = torch.load(pre_trained_model)#['model_state_dict']
chan = [10, 10]
side = [(53,4),( 31,4),(65,1)]
for j in range(3):
    weightMatrix = checkpoint['conv_blocks.' + str(j) + '.conv.weight']
    _min,_max=torch.min(weightMatrix),torch.max(weightMatrix)
    fig = plt.figure()
    data=np.array([_min+i*(_max-_min)/8 for i in range(9)])
    plt.imshow(data.reshape((3,3)), aspect='auto', vmin=_min, vmax=_max)
    plt.autoscale('False')
    plt.title(str(_min)+"_"+str(_max))
    fig.savefig('/home/pooja/PycharmProjects/lux_ai/outputs/weights/0_reference' + str(j) + '.png')
    plt.close(fig)
    for i in range(j):

        data = weightMatrix[i].flatten(start_dim=1, end_dim=-1)
        print(data.shape)
        data = data.reshape(side[j])
        fig = plt.figure()
        plt.imshow(data,aspect='auto', vmin = _min, vmax = _max)
        plt.autoscale('False')
        fig.savefig('/home/pooja/PycharmProjects/lux_ai/outputs/weights/conv'+str(j)+'_'+str(i)+'.png')
        plt.close(fig)