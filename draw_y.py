import numpy as np
from matplotlib import pyplot


def read_txt(url):
    # 打开文件并读取所有行
    with open(url, 'r') as file:
        lines = file.readlines()

    # 去除每行末尾的换行符并保存到列表中
    lines = [line.strip() for line in lines]

    print(lines)
    float_list = [float(item) for item in lines]
    print(float_list)
    return float_list

y_test=read_txt('result/3DCNN_y_test.txt')
y_pre_3DCNN=read_txt('result/3DCNN_y_pred.txt')
y_pre_3DCNNAtten=read_txt('result/3DCNNwithAtten_y_pred.txt')
y_pre_SVR=read_txt('result/SVR_y_pred.txt')
y_pre_Kriging=read_txt('result/Kriging_y_pred.txt')

y_test=np.array(y_test)
y_pre_3DCNN=np.array(y_pre_3DCNN)
y_pre_3DCNNAtten=np.array(y_pre_3DCNNAtten)
y_pre_SVR=np.array(y_pre_SVR)
y_pre_Kriging=np.array(y_pre_Kriging)
print('真实值的形状',y_test.shape)
print('3DCNN插值结果的形状',y_pre_3DCNN.shape)
print('3DCNNwithAtten插值结果的形状',y_pre_3DCNNAtten.shape)
print('SVR插值结果的形状',y_pre_SVR.shape)
print('Kriging插值结果的形状',y_pre_Kriging.shape)
# *115+7
y_test=y_test*115+7
y_pre_3DCNN=y_pre_3DCNN*115+7
y_pre_3DCNNAtten=y_pre_3DCNNAtten*115+7
y_pre_SVR=y_pre_SVR*115+7
y_pre_Kriging=y_pre_Kriging*115+7



# 将数组中的元素从float转换为int
y_test = y_test.astype(int)
y_pre_3DCNN = y_pre_3DCNN.astype(int)
y_pre_3DCNNAtten = y_pre_3DCNNAtten.astype(int)
y_pre_SVR = y_pre_SVR.astype(int)
y_pre_Kriging = y_pre_Kriging.astype(int)

# 使用unique去除重复元素，同时保留原始顺序
unique_elements, indices = np.unique(y_test, return_index=True)
# reconstructed_arr = y_test[np.sort(indices)]
y_test_del = y_test[indices]
y_pre_3DCNN_del = y_pre_3DCNN[indices]
y_pre_3DCNNAtten_del = y_pre_3DCNNAtten[indices]
y_pre_SVR_del = y_pre_SVR[indices]
y_pre_Kriging_del = y_pre_Kriging[indices]
print(y_test_del.shape,y_pre_3DCNN_del.shape,y_pre_3DCNNAtten_del.shape,y_pre_SVR_del.shape,y_pre_Kriging_del.shape)

import matplotlib.pyplot as plt
time=np.arange(8,40)


# 创建一个包含2x2子图的图形
fig, axes = plt.subplots(2, 2)

# 在每个子图中绘制内容
axes[0, 0].plot(time,y_test_del[8:40], color="blue",  linewidth=2.5, linestyle="-", label="ture")
axes[0, 0].plot(time, y_pre_3DCNN_del[8:40], color="green",  linewidth=2.5, linestyle="-", label="3DCNN")
axes[0, 0].set_title('3DCNN')
axes[0, 1].plot(time,y_test_del[8:40], color="blue",  linewidth=2.5, linestyle="-", label="ture")
axes[0, 1].plot(time,y_pre_3DCNNAtten_del[8:40], color="green",  linewidth=2.5, linestyle="-", label="ture")
axes[0, 1].set_title('3DCNN with CAM')
axes[1, 0].plot(time,y_test_del[8:40], color="blue",  linewidth=2.5, linestyle="-", label="ture")
axes[1, 0].plot(time,y_pre_SVR_del[8:40], color="green",  linewidth=2.5, linestyle="-", label="ture")
axes[1, 0].set_title('SVR')
axes[1, 1].plot(time,y_test_del[8:40], color="blue",  linewidth=2.5, linestyle="-", label="ture")
axes[1, 1].plot(time,y_pre_Kriging_del[8:40], color="green",  linewidth=2.5, linestyle="-", label="ture")
axes[1, 1].set_title('Kriging')
# 添加总标题
fig.suptitle('Result of 4 interpolation method', fontsize=14)

# 显示图形
plt.show()