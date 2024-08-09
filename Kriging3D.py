from datetime import datetime

from sklearn.metrics import r2_score
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import model_selection
from pykrige.ok3d import OrdinaryKriging3D

#读取每个层的钻孔数据
data_1=pd.read_excel(open('Data/data_1.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_2=pd.read_excel(open('Data/data_2.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_3=pd.read_excel(open('Data/data_3.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_4=pd.read_excel(open('Data/data_4.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_5=pd.read_excel(open('Data/data_5.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_6=pd.read_excel(open('Data/data_6.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_7=pd.read_excel(open('Data/data_7.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_8=pd.read_excel(open('Data/data_8.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_9=pd.read_excel(open('Data/data_9.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
data_10=pd.read_excel(open('Data/data_10.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
Y_1=pd.read_excel(open('Data/data_1.xlsx', 'rb'), usecols=[9])
Y_2=pd.read_excel(open('Data/data_2.xlsx', 'rb'), usecols=[9])
Y_3=pd.read_excel(open('Data/data_3.xlsx', 'rb'), usecols=[9])
Y_4=pd.read_excel(open('Data/data_4.xlsx', 'rb'), usecols=[9])
Y_5=pd.read_excel(open('Data/data_5.xlsx', 'rb'), usecols=[9])
Y_6=pd.read_excel(open('Data/data_6.xlsx', 'rb'), usecols=[9])
Y_7=pd.read_excel(open('Data/data_7.xlsx', 'rb'), usecols=[9])
Y_8=pd.read_excel(open('Data/data_8.xlsx', 'rb'), usecols=[9])
Y_9=pd.read_excel(open('Data/data_9.xlsx', 'rb'), usecols=[9])
Y_10=pd.read_excel(open('Data/data_10.xlsx', 'rb'), usecols=[9])
data_1=np.array(data_1)
data_2=np.array(data_2)
data_3=np.array(data_3)
data_4=np.array(data_4)
data_5=np.array(data_5)
data_6=np.array(data_6)
data_7=np.array(data_7)
data_8=np.array(data_8)
data_9=np.array(data_9)
data_10=np.array(data_10)
Y_1=np.array(Y_1)
Y_2=np.array(Y_2)
Y_3=np.array(Y_3)
Y_4=np.array(Y_4)
Y_5=np.array(Y_5)
Y_6=np.array(Y_6)
Y_7=np.array(Y_7)
Y_8=np.array(Y_8)
Y_9=np.array(Y_9)
Y_10=np.array(Y_10)
data=np.zeros([10, 13, 9])
label=np.zeros([10, 13, 1])
# 打印合并后的数组形状
#将数据合并为3DCNN所需的格式
data[0]=data_1
data[1]=data_2
data[2]=data_3
data[3]=data_4
data[4]=data_5
data[5]=data_6
data[6]=data_7
data[7]=data_8
data[8]=data_9
data[9]=data_10
label[0]=Y_1
label[1]=Y_2
label[2]=Y_3
label[3]=Y_4
label[4]=Y_5
label[5]=Y_6
label[6]=Y_7
label[7]=Y_8
label[8]=Y_9
label[9]=Y_10

print(data.shape)
print(label.shape)
data=data.reshape([130,9])
label=label.reshape([130,1])



X_test=np.load('result/X_test.npy')
y_test=np.load('result/y_test.npy')
print('测试集data形状;',X_test.shape)
print('测试集label形状;',y_test.shape)
del_index = np.array([7,343,19,120,15,22,3,10,17,4,1,21,63,54,18 ,8 , 43 , 12,
   6 , 52 , 40 , 80 , 47 , 59  , 0 ,239 , 11,  68 ,  5 , 57,  56  ,44 ,229,  82 , 45 , 60,
 144 , 38 , 36 , 26 ,234 ,231, 145 , 14 ,  2 , 61 , 73 , 49 , 37 , 65 ,149 , 53 , 41])
print(del_index.shape)
X_test_del=np.zeros([53,9])
y_test_del=np.zeros([53,1])
print('测试集data形状;',X_test_del.shape)
print('测试集label形状;',y_test_del.shape)
for i,j in zip(del_index,range(53)):
    print(i,j)
    X_test_del[j] = X_test[i]
    y_test_del[j] = y_test[i]

print('测试集data形状;',X_test_del.shape)
print('测试集label形状;',y_test_del.shape)




# print('1',X_test_del)
# print('2',y_test_del)
# for i in range(53):
#     for j in range(130):
#         if np.all(X_test_del[i]==Result_Fig[j]):
#             print(j)

index=np.array([99,
86,
112,
105,
108,
115,
95,
102,
110,
96,
93,
114,
69,
59,
111,
100,
47,
104,
98,
57,
44,
34,
51,
64,
91,
90,
103,
74,
97,
62,
61,
48,
79,
36,
49,
65,
13,
42,
39,
120,
84,
81,
14,
107,
94,
66,
27,
53,
40,
71,
19,
58,
45])
print(index.shape)



# 删除指定索引位置上的元素
X_train = np.delete(data, index, axis=0)
y_train = np.delete(label, index, axis=0)

# 输出删除元素后的新数组
print(X_train.shape)
print(y_train.shape)

X_test=X_test_del
y_test=y_test_del

start_time=datetime.now()
# ok3d = OrdinaryKriging3D(train_data[:, 0], train_data[:, 1], train_data[:, 2], train_data[:, 3], variogram_model="linear",verbose=1)
ok3d = OrdinaryKriging3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], y_train[:, 0], variogram_model="gaussian",verbose=1)
# variogram_model（str，可选） - 指定要使用的变异函数模型; 可能是以下之一：线性，幂，高斯，球形，指数，孔效应。 默认是线性变异函数模型。
end_time=datetime.now()
print(end_time-start_time)

#预测
y_pred, ss3d = ok3d.execute("points", X_test[:, 0], X_test[:, 1], X_test[:, 2])


print('真实结果形状',y_test.shape)
print('预测结果形状',y_pred.shape)
print('预测结果形状',y_pred)
# print(sum(Y_train[:, 0])/len(Y_train[:, 0]))
# 计算MSE
mean_squared_error = np.mean((y_test - y_pred) ** 2)

# 计算MAE
mean_absolute_error = np.mean(np.abs(y_test - y_pred))
R2 = r2_score(y_test, y_pred)

print('MSE:', mean_squared_error)
print('MAE:', mean_absolute_error)

print('R2:',R2)

np.savetxt('result/Kriging_y_test.txt', y_test, fmt='%.4f')
np.savetxt('result/Kriging_y_pred.txt', y_pred, fmt='%.4f')