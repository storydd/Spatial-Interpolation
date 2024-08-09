import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成随机数据
data=pd.read_excel(open('Data/address.xlsx', 'rb'), usecols=[9])
data=np.array(data)
y=np.zeros([13,10])
for i in range (13):
    y[i]=data[i*10:10+i*10].reshape([10])
# # 绘制群组箱形图
matplotlib.rcParams['font.family'] = 'Times New Roman'
f =plt.boxplot([y[0],y[1], y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9],y[10],y[11],y[12]],
            labels=['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5', 'Point 6', 'Point 7', 'Point 8', 'Point 9', 'Point 10', 'Point 11', 'Point 12', 'Point 13'],vert=True, sym='+b', showmeans=True,
                meanline=True, patch_artist=True, widths=0.2)
plt.title('Oil hydrocarbon Box-plot')
c_list = ['#ef476f', '#ffd166', '#ADEAEA', '#93DB70', '#ef476f', '#ffd166', '#ADEAEA', '#93DB70', '#ef476f', '#ffd166', '#ADEAEA', '#93DB70', '#ef476f']  # 颜色代码列表
for box, c in zip(f['boxes'], c_list):  # 对箱线图设置颜色
    box.set(color=c, linewidth=2)
    box.set(facecolor=c)
plt.show()
