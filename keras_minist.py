# 卷积神经网络手写数字识别

'''from keras import layers
from keras import models

# 开始构建卷积网络
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.summary()  # 可以看到滤波器在图像2维上的平移次数
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 获取训练数据，验证数据，测试数据
from keras.datasets import mnist
from keras.utils import to_categorical

(train_image, train_label), (test_image, test_label) = mnist.load_data()
train_image = train_image.reshape((60000, 28, 28, 1))
train_image = train_image.astype('float32') / 255

test_image = test_image.reshape((10000, 28, 28, 1))
test_image = test_image.astype('float32') / 255

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_image, train_label, epochs=3, batch_size=100)
test_loss, test_acc = model.evaluate(test_image, test_label)
print(test_loss, test_acc)
model.save('mnist_conv_model.h5')'''

from keras.models import load_model
import sys
import os
from PIL import Image  # cd到对应的安装目录下安装库
import numpy as np
model = load_model('mnist_conv_model.h5')
sys.path.append(os.pardir)  # 导入非本目录下的文件

image = Image.open("F:\Python_Projects\deeplearn\手画图.bmp")  # 用PIL中的Image.open打开图像
image_arr = np.array(image, dtype=int)  # 转化成numpy数组

# image_arr = image_arr.flatten()  # 化成一维，注意不能直接化，还需要赋值给他本身
image_arr = image_arr.reshape(1, 28, 28, 1)
print(image_arr.shape)
prd = model.predict(image_arr)  # 预测数字
np.set_printoptions(formatter={'float': '{:.4f}'.format})  # 可能的数字概率打印出来
print(prd)
print(np.argmax(prd))
