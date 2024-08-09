import numpy as np



# 读取一个形状为 (60, 66, 12, 9) 的示例数组
point = np.load('Result_Fig/point.npy')# (60, 66, 12, 9)
Level1=point[:,:,0:3]#(60, 66, 3, 9)
Level2=point[:,:,3:6]#(60, 66, 3, 9)
Level3=point[:,:,6:9]#(60, 66, 3, 9)
Level4=point[:,:,9:12]#(60, 66, 3, 9)
print(point.shape)
Level1_CNN=np.zeros([330,4,3,3,9])
Level2_CNN=np.zeros([330,4,3,3,9])
Level3_CNN=np.zeros([330,4,3,3,9])
Level4_CNN=np.zeros([330,4,3,3,9])
for i in range(15):
    for j in range(22):
        Level1_CNN[i*22+j,:,:,:,:]=Level1[0+i*4:4+i*4,0+j*3:3+j*3,:,:]
for i in range(15):
    for j in range(22):
        Level2_CNN[i*22+j,:,:,:,:]=Level2[0+i*4:4+i*4,0+j*3:3+j*3,:,:]
for i in range(15):
    for j in range(22):
        Level3_CNN[i*22+j,:,:,:,:]=Level3[0+i*4:4+i*4,0+j*3:3+j*3,:,:]
for i in range(15):
    for j in range(22):
        Level4_CNN[i*22+j,:,:,:,:]=Level4[0+i*4:4+i*4,0+j*3:3+j*3,:,:]
#
#
print('第一层的形状',Level1_CNN.shape)#(330,4,3,3,9)
print('第二层的形状',Level2_CNN.shape)#(330,4,3,3,9)
print('第三层的形状',Level3_CNN.shape)#(330,4,3,3,9)
print('第四层的形状',Level4_CNN.shape)#(330,4,3,3,9)
np.save('Result_Fig/level1.npy', Level1_CNN)
np.save('Result_Fig/level2.npy', Level2_CNN)
np.save('Result_Fig/level3.npy', Level3_CNN)
np.save('Result_Fig/level4.npy', Level4_CNN)