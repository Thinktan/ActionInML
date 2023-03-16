import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token

# print(" ".join([id_to_word[id_] for id_ in X_train[0][:]]))

import tensorflow_datasets as tfds
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

print(datasets.keys())
train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples

print(train_size, test_size)

for X_batch, y_batch in datasets["train"].batch(2).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review.decode("utf-8")[:200], "...")
        print("Label:", label, "= Positive" if label else "= Negative")
        print()

def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch



from collections import Counter

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    # print('----------------')
    x = 0
    for review in X_batch:
        # print("==========================")
        # print(x)
        x+=1
        # print(review.numpy())
        vocabulary.update(list(review.numpy()))

print(vocabulary.most_common()[:3])
print(len(vocabulary))

vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]

print(truncated_vocabulary)




