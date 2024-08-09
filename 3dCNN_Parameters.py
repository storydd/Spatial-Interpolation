from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten,Activation,GlobalAveragePooling3D, Layer,Conv3DTranspose,Layer, GlobalAveragePooling3D, Reshape, Dense, Multiply
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import Constant
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
from opt_einsum.backends import torch
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

#读取每个层的钻孔数据
data_pd=pd.read_excel(open('Data/Data.xlsx', 'rb'), usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
Y_pd=pd.read_excel(open('Data/Data.xlsx', 'rb'), usecols=[9])
data_np=np.array(data_pd)
Y_np=np.array(Y_pd)

data=np.zeros([156, 14, 9])
label=np.zeros([156, 14, 1])
# 将数据合并为3DCNN所需的格式
for i in range(156):
    for j in range(14):
        data[i,j]=data_np[i*14+j]
for i in range(156):
    for j in range(14):
        label[i,j]=Y_np[i*14+j]


data_cnn=np.zeros([13,12,14,9])
label_cnn=np.zeros([13,12,14,1])

#整理input
for i in range(13):
    data_cnn[i]=data[0+i*12:12+i*12,:,:]
for i in range(13):
    label_cnn[i]=label[0+i*12:12+i*12,:,:]

data_cnn = np.transpose(data_cnn, (0, 2, 1, 3))#(13, 14, 12, 9)
label_cnn = np.transpose(label_cnn, (0, 2, 1, 3))#(13, 14, 12, 1)
data_3dCNN=np.zeros([13, 14, 3, 4, 9])
label_3dCNN=np.zeros([13, 14, 3, 4, 1])
#整理input
for i in range(13):
    for j in range(3):
        data_3dCNN[i,:,j]=data_cnn[i,:,0+j*4:4+j*4]

for i in range(13):
    for j in range(3):
        label_3dCNN[i,:,j]=label_cnn[i,:,0+j*4:4+j*4]

data_3dCNN = np.transpose(data_3dCNN, (0, 1, 3, 2,4))#(13, 14, 4, 3, 1)
label_3dCNN = np.transpose(label_3dCNN, (0, 1, 3, 2,4))#(13, 14, 4, 3, 1)
fin_data=np.zeros([156, 3, 4, 3, 9])
fin_label=np.zeros([156, 3, 4, 3, 1])
#整理input
for i in range(13):
    for j in range(12):
        fin_data[i*12+j]=data_3dCNN[i,0+j:3+j]

#整理output
for i in range(13):
    for j in range(12):
        fin_label[i*12+j]=label_3dCNN[i,0+j:3+j]


# # #
# X_train, X_valtest, y_train, y_valtest = model_selection.train_test_split(fin_data, fin_label, test_size = 0.1, random_state = 1)
# X_val, X_test, y_val, y_test = model_selection.train_test_split(X_valtest, y_valtest, test_size = 0.5, random_state = 1)
# print('训练集数据个数',X_train.shape)
# print('验证集数据个数',X_val.shape)
# print('测试集数据个数',X_test.shape)
# #
# #
# #
# X_train=X_train.reshape([5040,9])
# X_val=X_val.reshape([288,9])
# X_test=X_test.reshape([288,9])
# X_scaler = MinMaxScaler()
# X_scaler.fit(X_train)
# X_train=X_scaler.transform(X_train)
# X_val=X_scaler.transform(X_val)
# X_test=X_scaler.transform(X_test)
# print(X_scaler.data_max_)
# print(X_scaler.data_min_)
# X_train=X_train.reshape([140, 3, 4, 3, 9])
# X_val=X_val.reshape([8, 3, 4, 3, 9])
# X_test=X_test.reshape([8, 3, 4, 3, 9])
# joblib.dump(X_scaler, 'X_scalar')
# #
# #
# #
# y_train=y_train.reshape([5040,1])
# y_val=y_val.reshape([288,1])
# y_test=y_test.reshape([288,1])
# y_scaler = MinMaxScaler()
# y_scaler.fit(y_train)
# y_train=y_scaler.transform(y_train)
# y_val=y_scaler.transform(y_val)
# y_test=y_scaler.transform(y_test)
# print(y_scaler.data_max_)
# print(y_scaler.data_min_)
# y_train=y_train.reshape([140, 3, 4, 3, 1])
# y_val=y_val.reshape([8, 3, 4, 3, 1])
# y_test=y_test.reshape([8, 3, 4, 3, 1])
# joblib.dump(y_scaler, 'y_scalar')

X=fin_data.reshape([5616,9])
X_scaler = MinMaxScaler()
X_scaler.fit(X)
X=X_scaler.transform(X)
print(X_scaler.data_max_)
print(X_scaler.data_min_)
X=X.reshape([156, 3, 4, 3, 9])
joblib.dump(X_scaler, 'X_all_scalar')
#
#
#
y=fin_label.reshape([5616,1])
y_scaler = MinMaxScaler()
y_scaler.fit(y)
y=y_scaler.transform(y)
print(y_scaler.data_max_)
print(y_scaler.data_min_)
y=y.reshape([156, 3, 4, 3, 1])

joblib.dump(y_scaler, 'y_all_scalar')




class ChannelAttention(Layer):
    def __init__(self, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        print('input_shape',input_shape)
        self.shared = Dense(int(input_shape[-1] // self.reduction_ratio), activation='relu', kernel_initializer='he_normal')
        self.scales = Dense(int(input_shape[-1] // self.reduction_ratio), activation='sigmoid', kernel_initializer='he_normal')

    class ChannelAttention(Layer):
        def __init__(self, reduction_ratio=8, **kwargs):
            super(ChannelAttention, self).__init__(**kwargs)
            self.reduction_ratio = reduction_ratio

        def build(self, input_shape):
            channels = input_shape[-1]
            self.shared = GlobalAveragePooling3D(data_format="channels_last")
            self.dense1 = Dense(channels // self.reduction_ratio, activation='relu')
            self.dense2 = Dense(channels, activation='sigmoid')

        def call(self, inputs):
            pool = self.shared(inputs)
            reshape = Reshape((1, 1, 1, -1))(pool)
            dense1_output = self.dense1(reshape)
            dense2_output = self.dense2(dense1_output)
            scales = Reshape((1, 1, 1, -1))(dense2_output)
            output = Multiply()([inputs, scales])
            return output



# #3DCNN的输入为（3，4，3，9）,3:层数(通道数)，4,3：将每层的12个点位变为4×3,9：每个点位的特征数
def create_model():
    model = Sequential()
    model.add(Conv3D(data_format='channels_last',filters=16, kernel_size=(2,2,2), strides=(1, 1, 1), input_shape=(3, 4, 3, 9)))
    model.add(Activation('tanh'))
    model.add(ChannelAttention())  # 在这里使用通道注意力机制
    model.add(Conv3D(data_format='channels_last',filters=32, kernel_size=(2,2,2), strides=(1, 1, 1)))
    model.add(Activation('tanh'))
    model.add(ChannelAttention())   # 在这里使用通道注意力机制
    model.add(Conv3D(data_format='channels_last', filters=64, kernel_size=(1, 2, 1), strides=(1, 1, 1)))
    model.add(Activation('tanh'))
    model.add(ChannelAttention())  # 在这里使用通道注意力机制
    model.add(Conv3DTranspose(data_format='channels_last', filters=32, kernel_size=(1, 2, 1), strides=(1, 1, 1)))
    model.add(Activation('tanh'))
    model.add(Conv3DTranspose(data_format='channels_last',filters=16, kernel_size=(1, 2, 2), strides=(1, 1, 1)))
    model.add(Activation('tanh'))
    model.add(Conv3DTranspose(data_format='channels_last',filters=8, kernel_size=(1, 2, 2), strides=(1, 1, 1)))
    model.add(Activation('tanh'))
    model.add(Conv3DTranspose(data_format='channels_last',filters=1, kernel_size=(3, 1, 1), strides=(1, 1, 1)))
    model.add(Activation('tanh'))
    # 打印模型摘要
    model.summary()
    initial_learning_rate = 0.01
    decay_steps = 40
    decay_rate = 0.9
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model.compile(optimizer = optimizer, loss = "mse", metrics=["mae"])
    return model
print("create model and train model")
model = create_model()
start_time=datetime.now()

history = model.fit(X, y, epochs=400, verbose=1)#validation_data=(X_val, y_val)
# end_time=datetime.now()
# print('time:',end_time-start_time)
# print('model evaluation: ', model.evaluate(X_test, y_test))
# y_pred=model.predict(X_test)
# y_test=y_test.reshape([-1,1])
# y_pred=y_pred.reshape([-1,1])
# R2 = r2_score(y_test, y_pred)
#
# print('R2:',R2)
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.xlabel('Epochs', fontsize=12)
# pyplot.ylabel('Loss', fontsize=12)
# pyplot.show()
# 保存整个模型
model.save('model/model8')