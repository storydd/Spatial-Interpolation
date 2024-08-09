import random
import numpy as np
i=0
j=0
point=np.zeros([60,66,12,9])
feature=np.zeros([9])
for x in np.arange(4009695,4009755,1):
    j = 0
    for y in np.arange(40533713,40533779,1):
        for z in np.arange(0,6,0.5):
            if (z<=1.5):
                feature = [x, y, z, 7.68, 1.43,7.5, 0.83,0.14,0.4]
                point[i, j, int(z/0.5),:]=feature
            if (z>1.5) & (z<=3.5):
                feature = [x, y, z, 7.7, 1.46, 15, 0.41,0.27,0.4]
                point[i, j, int(z/0.5), :] = feature
            if (z>3.5) & (z<=5):
                feature = [x, y, z, 7.61, 1.53,10,0.36,0.26,0.1]
                point[i, j, int(z/0.5), :] = feature
            if (z>5):
                feature = [x, y, z, 7.74, 1.62,0.02,0.17,0.02,0.1]
                point[i, j, int(z/0.5), :] = feature
        print(i, j)
        j = j + 1
    i = i + 1
np.save('Result_Fig/point.npy', point)