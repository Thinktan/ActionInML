
import tensorflow as tf
from Util13 import PROJECT_ROOT_DIR
import os

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))


filepath = os.path.join(PROJECT_ROOT_DIR, "test", "my_contacts.tfrecord")
with tf.io.TFRecordWriter(filepath) as f:
    f.write(person_example.SerializeToString())

feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

for serialized_example in tf.data.TFRecordDataset([filepath]):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

print(parsed_example)
print(parsed_example["emails"].values[0])