import numpy as np

import pandas as pd

# 读取xlsx文件的前508行
data=np.load('Result_Fig\point.npy')
level1=data[0:4,0:3,0:3,:]
print(level1[:,0,:,:].shape)
print(level1[:,0,:,:])
