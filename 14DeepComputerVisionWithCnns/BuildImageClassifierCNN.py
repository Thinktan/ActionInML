
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()

# 卷积层和最大池化层
model.add(keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))

# --> shape(4, 4, 64)

# 扁平层
model.add(keras.layers.Flatten())

# 全连接层分类输出
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


print(model.summary())