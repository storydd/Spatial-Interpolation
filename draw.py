import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.tri as mtri
# 读取xlsx文件的前508行
df = pd.read_excel('Result_Fig/result.xlsx')
data=np.array(df)
print(data.shape)
# 坐标点的信息和值
coordinates = data[:,0:3]
print(coordinates.shape)
coordinates[:,2]=coordinates[:,2]*(-1)
values = data[:,3]  # 对应每个坐标点的值

# 创建一个3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取坐标点的x、y和z值
x = [point[0] for point in coordinates]
y = [point[1] for point in coordinates]
z = [point[2] for point in coordinates]

# 绘制3D散点图，并设置更明显的颜色
scatter = ax.scatter(x, y, z, c=values, cmap='jet', vmin=min(values), vmax=max(values))

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加颜色映射条
cbar = fig.colorbar(scatter)
cbar.set_label('Value')

# 显示图形
plt.show()