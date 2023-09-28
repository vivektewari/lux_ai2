import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
ref=1
position=[i for i in range(12)]#(1,2)
#position=(4,10)
excel1,excel2=0,60#430,400#type.reverse
skippers=[2,4,0]#[1,2,4,7,8,9,10,11,12,13,14,15,16,17,18,19]
if ref==1 :
    actions=19
    initial_skip=0
    index_start = 0
elif ref==2 :
    actions=4
    initial_skip = 15*19
    index_start=36
action_frame1=[]
action_frame2=[]
act_probs1=[]
act_probs2=[]
action_sel=[]
entropy =np.array([0.0,0.0])

for i in range(actions):

    action_frame1.append(pd.read_excel('//home//pooja//PycharmProjects//lux_ai//outputs//tracking//ct_build///output_trials/tracking//sel_act_'+str(excel1)+'.xlsx',
                    skiprows=(initial_skip+i*(15)),nrows=12,index_col=0).to_numpy()[:,:12])
    action_frame2.append(pd.read_excel('//home//pooja//PycharmProjects//lux_ai//outputs//tracking//ct_build//output_trials//tracking//sel_act_'+str(excel2)+'.xlsx',
                       skiprows=initial_skip + i * (15), nrows=12, index_col=0).to_numpy()[:,:12])
    if i not in skippers:
        act_probs1.append(action_frame1[i][position].sum())
        act_probs2.append(action_frame2[i][position].sum())
        action_sel.append(str(i))
        #entropy+=np.array([action_frame1[i][position]*np.log(action_frame1[i][position]),action_frame2[i][position]*np.log(action_frame2[i][position])])
        #action_sel2.append(str(i+0.00001))
per=[act_probs1,act_probs2]
for i in range(2):
    factor=1#np.sum(np.array(per[i]))
    for j in range(len(act_probs1)):
        per[i][j]=per[i][j]/factor
        entropy[i] += np.array([per[i][j]*np.log(per[i][j])])

barWidth=0.2
X_axis =np.arange(len(action_sel))
X_axis2=[x + barWidth for x in X_axis]
plt.bar(X_axis ,per[0],width=barWidth,label=1)
plt.bar(X_axis2,per[1],width=barWidth,label=2)
plt.title(str(np.array(per[0]).sum())+'_'+str(np.array(per[1]).sum())+'_'+
          str(entropy[0])+'_'+str(entropy[1]))
plt.xticks([r + barWidth for r in range(len(action_sel))],
        action_sel)
plt.legend()
plt.show()

