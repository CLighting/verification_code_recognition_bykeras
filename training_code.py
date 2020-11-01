# 使用keras写CNN

# 导入必要的包
import numpy as np
from keras.optimizers import Adam

from keras.utils import np_utils
from keras.models import Sequential # 把器具搞出来
from keras.layers import Dense # 全连接层，将激活函数应用于输出,二维卷积。

# 准备数据

# #加载数据
train_data = np.load('verification_code_img&label/train.npy')
train_label = np.load('verification_code_img&label/train_label.npy')
test_data = np.load('verification_code_img&label/test.npy')
test_label = np.load('verification_code_img&label/test_label.npy')

# #处理数据

# ##将图像像素值从[0,255]规格化到[-0.5,0.5],使用较小的居中值通常会得到更好的结果
train_data = (train_data / 255) - 0.5
test_data = (test_data / 255) - 0.5

# ##把二维的变成三维的，keras需要三维
train_data = np.expand_dims(train_data, axis=3)
test_data = np.expand_dims(test_data, axis=3)

# ##再把数据展平(60x160)
train_data = train_data.reshape((-1, 9600))
test_data = test_data.reshape((-1, 9600))

# ##处理标签集(4,3,62)
# 造映射字典
labeldict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
             'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
             'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
labeldict_a = []
labeldict_n = [str(x) for x in range(10)]
for i in labeldict.keys():
    labeldict_a.append(i.lower())
num_a = list(range(26, 52))
num_n = list(range(52, 62))
labeldict_a = dict(zip(labeldict_a, num_a))
labeldict_n = dict(zip(labeldict_n, num_n))
labeldict.update(labeldict_a)
labeldict.update(labeldict_n)
# 处理标签
# 训练标签
train_label1 = []
test_label1 = []
for label_num in range(len(train_label)):
  c0 = np_utils.to_categorical(labeldict[train_label[label_num][0]], 62)
  c1 = np_utils.to_categorical(labeldict[train_label[label_num][1]], 62)
  c2 = np_utils.to_categorical(labeldict[train_label[label_num][2]], 62)
  c3 = np_utils.to_categorical(labeldict[train_label[label_num][3]], 62)
  c = np.concatenate((c0, c1, c2, c3), axis=0)
  train_label1.append(c)
# 测试标签
for label_num1 in range(len(test_label)):
  c0 = np_utils.to_categorical(labeldict[test_label[label_num1][0]], 62)
  c1 = np_utils.to_categorical(labeldict[test_label[label_num1][1]], 62)
  c2 = np_utils.to_categorical(labeldict[test_label[label_num1][2]], 62)
  c3 = np_utils.to_categorical(labeldict[test_label[label_num1][3]], 62)
  c = np.concatenate((c0, c1, c2, c3), axis=0)
  test_label1.append(c)

test_label1 = np.array(test_label1)
train_label1 = np.array(train_label1)
# 构建模型
model = Sequential([
  Dense(128, activation='relu', input_shape=(9600,)),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(248, activation='sigmoid'),
])

# 配置模型
model.compile(
  optimizer=Adam(lr=0.01),
  loss='binary_crossentropy',
  metrics=['accuracy'],
)

# 训练模型
model.fit(
  train_data,
  train_label1,
  epochs=5,
  batch_size=32,
)

# 评价模型
loss, acc = model.evaluate(
  test_data,
  test_label1
)
print(loss, acc)
# 保存模型
model.save('verification_code_recognition_model.h5')