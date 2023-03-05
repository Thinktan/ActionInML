from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt
import tensorflow as tf

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

img = load_sample_images()["images"][0]
# plt.imshow(img)
# plt.axis("off")
# plt.title("Original Image")
# plt.show()

data = tf.io.encode_jpeg(img)
example_with_image = Example(
    features=Features(
        feature={"image": Feature(bytes_list=BytesList(value=[data.numpy()]))}
    )
)

serialized_example = example_with_image.SerializeToString()

# then save to TFRecord

feature_description = { "image": tf.io.VarLenFeature(tf.string) }
example_with_image = tf.io.parse_single_example(serialized_example, feature_description)
decoded_img = tf.io.decode_jpeg(example_with_image["image"].values[0])

plt.imshow(decoded_img)
plt.title("Decoded Image")
plt.axis("off")
plt.show()