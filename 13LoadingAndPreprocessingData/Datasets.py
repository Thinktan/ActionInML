import tensorflow as tf

X = tf.range(10)
print(X)

dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)

print("===================")
for item in dataset:
    print(item)

print("===================")

dataset = dataset.repeat(3)
print(dataset)
for item in dataset:
    print(item)

print("===================")

dataset = dataset.batch(7)
for item in dataset:
    print(item)

print("===================")

dataset = dataset.map(lambda x: x*2)
for item in dataset:
    print(item)

print("===================")

dataset = dataset.unbatch()
print(dataset)

for item in dataset:
    print(item)

print("===================")


dataset = dataset.filter(lambda x: x < 10)
for item in dataset:
    print(item)

print("===================")

tf.random.set_seed(42)
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
for item in dataset:
    print(item)
















