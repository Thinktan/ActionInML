import os.path

import tensorflow as tf
from Util13 import PROJECT_ROOT_DIR

filepath = os.path.join(PROJECT_ROOT_DIR, "test", "my_data.tfrecord")

# with tf.io.TFRecordWriter(filepath) as f:
#     f.write(b"This is the first record")
#     f.write(b"And this is the second record")


# filepaths = [filepath]
# dataset = tf.data.TFRecordDataset(filepaths)
# for item in dataset:
#     print(item)


# 压缩的TFRecord
options = tf.io.TFRecordOptions(compression_type="GZIP")
filepath2 = os.path.join( PROJECT_ROOT_DIR, "test", "my_compressed.tfrecord" )
with tf.io.TFRecordWriter(filepath2, options) as f:
    f.write(b"I love you, just like you love me")

filepaths2 = [filepath2]
dataset = tf.data.TFRecordDataset(filepaths2, compression_type="GZIP")

for item in dataset:
    print(item)
