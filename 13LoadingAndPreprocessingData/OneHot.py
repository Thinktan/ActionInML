import tensorflow as tf


vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_ovv_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_ovv_buckets)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)

cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab) + num_ovv_buckets)
print(cat_one_hot)


