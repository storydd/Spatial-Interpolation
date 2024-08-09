from datetime import datetime

import numpy as np
import pandas as pd
import h5py
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.python.keras.layers import Conv3DTranspose
from sklearn import model_selection
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

#[10,13,9]代表10个不同层次土壤，每层有13个点位，每个点位有9个特征
#[4,3,9*3]从13个点位中选取12个变为4*3的二维矩阵，代表同一层不同点位，将3个层堆叠代替3DCNN的时间维度
import numpy as np

# 使用transpose()函数进行转换
data_transposed = np.transpose(data, (0, 1, 2))
label_transposed = np.transpose(label, (0, 1, 2))
# 打印转换后的数组形状
print(data_transposed.shape)  # 输出：(10, 13, 9)
print(label_transposed.shape)  # 输出：(10, 13, 1)
tdcnn_input=np.zeros([104,3,4,3,9])
tdcnn_output=np.zeros([104,3,4,3,1])


#整理input
for i in range(13):
    # 使用切片操作去除第三个向量并得到形状为 (12, 9) 的数组
    data_del = np.delete(data_transposed, i, axis=1)
    # data_del = np.transpose(data_del, (1, 2, 0))
    print('data_del的形状', data_del.shape)  # (10,12,9)
    mid_data= np.zeros([8,3,4,3,9])
    for j in range(8):
        data_chose3=data_del[j:j+3,:,:]#(3, 12, 9)
        print('第一次data_chose3的形状',data_chose3.shape)
        data_chose3=data_chose3.reshape([3, 4, 3, 9])#(3, 4, 3, 9)
        print('第二次data_chose3的形状',data_chose3.shape)
        mid_data[j]=data_chose3
    print('mid_data的形状',mid_data.shape)
    tdcnn_input[i*8:(i+1)*8,:,:]=mid_data


#整理output
for i in range(13):
    # 使用切片操作去除第三个向量并得到形状为 (12, 9) 的数组
    label_del = np.delete(label_transposed, i, axis=1)
    # label_del = np.transpose(label_del, (1, 2, 0))
    print('label_del的形状', label_del.shape)  # (10, 1, 12)
    mid_label= np.zeros([8,3,4,3,1])
    for j in range(8):
        label_chose3=label_del[j:j+3,:,:]#(1, 3, 12)
        print('第一次label_chose3的形状',label_chose3.shape)
        label_chose3=label_chose3.reshape([3, 4, 3, 1])#(1, 3, 4, 3)
        print('第二次label_chose3的形状',label_chose3.shape)
        mid_label[j]=label_chose3
    print('mid_label的形状',mid_label.shape)
    tdcnn_output[i*8:(i+1)*8,:,:]=mid_label

print('输入：',tdcnn_input.shape)
print('输出：',tdcnn_output.shape)
#
#
#
#

# #3DCNN的输入为（3, 4, 3, 9）,3:层数(通道数)，4,3：每层点位数，9：每层特征数
def create_model():
    model = Sequential()
    model.add(Conv3D(data_format='channels_last',filters=16, kernel_size=(2,2,2), strides=(1, 1, 1), input_shape=(3, 4, 3, 9)))
    print(model.output_shape)  # 输出第一层卷积后的数据形状
    model.add(Activation('relu'))
    model.add(Conv3D(data_format='channels_last',filters=32, kernel_size=(2,2,2), strides=(1, 1, 1)))
    print(model.output_shape)  # 输出第二层卷积后的数据形状
    model.add(Activation('relu'))
    model.add(Conv3DTranspose(data_format='channels_last',filters=16, kernel_size=(1, 2, 2), strides=(1, 1, 1)))
    print(model.output_shape)  # 输出第一层转置卷积后的数据形状
    model.add(Conv3DTranspose(data_format='channels_last',filters=8, kernel_size=(1, 2, 2), strides=(1, 1, 1)))
    print(model.output_shape)  # 输出第二层卷转置积后的数据形状
    model.add(Conv3DTranspose(data_format='channels_last',filters=1, kernel_size=(3, 1, 1), strides=(1, 1, 1)))
    print(model.output_shape)  # 输出第三层转置卷积后的数据形状
    # 打印模型摘要
    model.summary()
    model.compile(optimizer = 'adam', loss = "mse", metrics=["mae"])
    return model
#
#
#
#
# n_split = 5
# avg_loss = [0, 0]
# min_loss = 1
# for train_index, test_index in KFold(n_split, shuffle=True, random_state=1).split(tdcnn_input):
#     print("test index: ", test_index)
#     x_train, x_test = tdcnn_input[train_index], tdcnn_input[test_index]
#     y_train, y_test = tdcnn_output[train_index], tdcnn_output[test_index]
#
#     print("create model and train model")
#     print('训练集data形状:', x_train.shape)
#     print('训练集label形状:', y_train.shape)
#     model = create_model()
#
#     history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=400, verbose=1)
#
#     print('model evaluation: ', model.evaluate(x_test, y_test))
#     #         if model.evaluate(x_test, y_test)<min_loss:
#     #             min_loss= model.evaluate(x_test, y_test)
#     #             model.save('my_twotanh_model.h5')
#     avg_loss = avg_loss + model.evaluate(x_test, y_test)
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.xlabel('Epochs', fontsize=12)
#     pyplot.ylabel('Loss', fontsize=12)
#     pyplot.show()

X_train, X_valtest, y_train, y_valtest = model_selection.train_test_split(tdcnn_input, tdcnn_output, test_size = 0.3, random_state = 12)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_valtest, y_valtest, test_size = 0.3, random_state = 12)
print('训练集data形状;',X_train.shape)
print('训练集label形状;',y_train.shape)
print('验证集data形状;',X_val.shape)
print('验证集label形状;',y_val.shape)
print('测试集data形状;',X_test.shape)
print('测试集label形状;',y_test.shape)


from sklearn.metrics import r2_score

print("create model and train model")
model = create_model()
start_time=datetime.now()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=400, verbose=1)
end_time=datetime.now()
print('time:',end_time-start_time)
print('model evaluation: ', model.evaluate(X_test, y_test))
#     #         if model.evaluate(x_test, y_test)<min_loss:
#     #             min_loss= model.evaluate(x_test, y_test)
#     #             model.save('my_twotanh_model.h5')
#     avg_loss = avg_loss + model.evaluate(x_test, y_test)
y_pred=model.predict(X_test)
y_test=y_test.reshape([-1,1])
y_pred=y_pred.reshape([-1,1])
R2 = r2_score(y_test, y_pred)
print('R2:',R2)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize=12)
pyplot.ylabel('Loss', fontsize=12)
pyplot.show()

np.savetxt('result/3DCNN_y_test.txt', y_test, fmt='%.4f')
np.savetxt('result/3DCNN_y_pred.txt', y_pred, fmt='%.4f')



