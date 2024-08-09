import joblib
import keras.models
import numpy as np
import pandas as pd
from joblib import  load
# 加载模型
model=keras.models.load_model('model/model8')
# # 加载数据
level1=np.load('Result_Fig/level1.npy')#(330, 4, 3, 3, 9)
level2=np.load('Result_Fig/level2.npy')#(330, 4, 3, 3, 9)
level3=np.load('Result_Fig/level3.npy')#(330, 4, 3, 3, 9)
level4=np.load('Result_Fig/level4.npy')#(330, 4, 3, 3, 9)

level1 = level1.transpose((0, 3, 1, 2, 4))#(330, 3, 4, 3, 9)
level2 = level2.transpose((0, 3, 1, 2, 4))#(330, 3, 4, 3, 9)
level3 = level3.transpose((0, 3, 1, 2, 4))#(330, 3, 4, 3, 9)
level4 = level4.transpose((0, 3, 1, 2, 4))#(330, 3, 4, 3, 9)
level1=level1.reshape([11880,9])
level2=level2.reshape([11880,9])
level3=level3.reshape([11880,9])
level4=level4.reshape([11880,9])
print(np.min(level1[:,2]))
X_scaler = joblib.load('X_all_scalar')
print(X_scaler.data_max_)
print(X_scaler.data_min_)


level1=X_scaler.transform(level1)
level2=X_scaler.transform(level2)
level3=X_scaler.transform(level3)
level4=X_scaler.transform(level4)

level1=level1.reshape([330, 3, 4, 3, 9])
level2=level2.reshape([330, 3, 4, 3, 9])
level3=level3.reshape([330, 3, 4, 3, 9])
level4=level4.reshape([330, 3, 4, 3, 9])
level1_result=model.predict(level1)#(1551, 3, 4, 3, 1)
level2_result=model.predict(level2)#(1551, 3, 4, 3, 1)
level3_result=model.predict(level3)#(1551, 3, 4, 3, 1)
level4_result=model.predict(level4)#(1551, 3, 4, 3, 1)

level1=level1.reshape([11880,9])
level2=level2.reshape([11880,9])
level3=level3.reshape([11880,9])
level4=level4.reshape([11880,9])

print(np.min(level1[:,2]))


level1=X_scaler.inverse_transform(level1)
level2=X_scaler.inverse_transform(level2)
level3=X_scaler.inverse_transform(level3)
level4=X_scaler.inverse_transform(level4)

print(np.min(level1[:,2]))

level1=level1.reshape([330, 3, 4, 3, 9])
level2=level2.reshape([330, 3, 4, 3, 9])
level3=level3.reshape([330, 3, 4, 3, 9])
level4=level4.reshape([330, 3, 4, 3, 9])



y_scaler = joblib.load('y_all_scalar')

level1_result=level1_result.reshape([11880,1])
level2_result=level2_result.reshape([11880,1])
level3_result=level3_result.reshape([11880,1])
level4_result=level4_result.reshape([11880,1])

level1_result=y_scaler.inverse_transform(level1_result)
level2_result=y_scaler.inverse_transform(level2_result)
level3_result=y_scaler.inverse_transform(level3_result)
level4_result=y_scaler.inverse_transform(level4_result)

level1_result=level1_result.reshape([330, 3, 4, 3,1])
level2_result=level2_result.reshape([330, 3, 4, 3,1])
level3_result=level3_result.reshape([330, 3, 4, 3,1])
level4_result=level4_result.reshape([330, 3, 4, 3,1])

level1_finall= np.concatenate((level1, level1_result), axis=4)#(229392, 11)
level2_finall= np.concatenate((level2, level2_result), axis=4)#(229392, 11)
level3_finall= np.concatenate((level3, level3_result), axis=4)#(229392, 11)
level4_finall= np.concatenate((level4, level4_result), axis=4)#(229392, 11)
level1_finall=level1_finall.reshape([11880,10])
level2_finall=level2_finall.reshape([11880,10])
level3_finall=level3_finall.reshape([11880,10])
level4_finall=level4_finall.reshape([11880,10])
del level1,level2,level3,level4,level1_result,level2_result,level3_result,level4_result
finall_mid=np.concatenate((level1_finall, level2_finall), axis=0)
finall_mid1=np.concatenate((finall_mid, level3_finall), axis=0)
finall=np.concatenate((finall_mid1, level4_finall), axis=0)
print('最终结果',finall.shape)




# 创建 DataFrame
df1 = pd.DataFrame(finall)



# 设置显示格式为不使用科学计数法
pd.set_option('display.float_format', lambda x: '%.6f' % x)
#
# # 将 DataFrame 写入 Excel 文件
df1.to_excel('Result_Fig/result.xlsx', index=False, header=False)
# X坐标	Y坐标	z坐标(深度)	PH	湿容重g/cm-3	Kv/m·d-1	有机碳/g·kg-1	平均给水度	盐度(dS/m)	石油烃指数