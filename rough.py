import numpy as np

p=np.array([[0.8,0.2],[0.1,0.9]])
p_=np.array([0.5,0.5])
for i in range(100):
    p_=np.matmul(p_,p)
print(p_)