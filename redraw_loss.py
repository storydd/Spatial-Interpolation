import matplotlib
import numpy as np
import matplotlib.pyplot as plt
train_loss=[]
val_loss=[]
print(type(train_loss))
# train_loss=np.zeros([400])
# val_loss=np.zeros([400])
with open('result/3DCNN_noAtten_2.txt', 'r') as file:
    for line in file:
        print(line)
        print(line[6:12])
        print(line[39:45])
        train_loss.append(float(line[6:12]))
        val_loss.append(float(line[39:45]))
print(len(train_loss))
print(len(val_loss))




# 创建 x 轴坐标（训练步骤）
steps = range(1, 401)
# 绘制损失图
# 设置字体样式为Arial
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.plot(steps, train_loss,c=(0.5, 0.8, 0.9),ls='-',label='train_Loss')
plt.plot(steps, val_loss,c=(0.9, 0.5, 0.5),ls='-',label='val_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# 添加图例
plt.legend()

# 设置y轴范围
plt.ylim(0, 0.02)
plt.show()