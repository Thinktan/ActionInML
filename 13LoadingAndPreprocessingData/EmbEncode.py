import tensorflow as tf

vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

embedding_dim = 2
num_ovv_buckets = 2
embed_init = tf.random.uniform([len(vocab) + num_ovv_buckets, embedding_dim])
print(embed_init)
print("======")
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)
print("======")

from tensorflow import keras

indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_ovv_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_ovv_buckets)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)


print(tf.nn.embedding_lookup(embedding_matrix, cat_indices))

#embedding = keras.layers.Embedding(input_dim=len(vocab)+num_ovv_buckets, output_dim=embedding_dim)
