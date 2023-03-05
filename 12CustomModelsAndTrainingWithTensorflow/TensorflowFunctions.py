import Util12
import tensorflow as tf
from tensorflow import keras

#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



def cube(x):
    return x**3

print(cube(2))
print(cube(tf.constant(2.0)))

tf_cube = tf.function(cube)
print(tf_cube)

print(tf_cube(2))
print(tf_cube(tf.constant(2.0)))