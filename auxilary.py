
import numpy as np
from visdom import Visdom
import copy
class VisdomLinePlotter(object):
    def __init__(self, env_name='default'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        #self.viz.delete_env(env_name)


    """Plots to Visdom"""
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:

            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]),win=var_name, env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=var_name, name=split_name, update = 'append')

def visdom_print(visualizer, epoch,key_case,log1,log2):

    logs=copy.deepcopy(log1)
    for key in key_case:
        logs[key]=[]
        for i in range(len(log1[key])):
            logs[key].append([log1[key][i][0],log2[key][i][0]])


    for key in logs.keys():
        arr = np.array(logs[key])

        if arr.shape[1] == 2:  # makinf sure there are two playesrs only
            avg = np.mean(arr, axis=0)
            for i in range(2):
                visualizer.plot(key, str(i), key, epoch, avg[i])
        else:
            visualizer.plot(key, '0', key, epoch, np.mean(arr))
    return  visualizer
