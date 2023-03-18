import numpy as np
from tensorflow import keras
import tensorflow as tf
from Util17 import generate_3d_data

np.random.seed(4)

X_train = generate_3d_data(60)
X_train = X_train - X_train.mean(axis=0, keepdims=0)

# print(X_train)

np.random.seed(42)
tf.random.set_seed(42)

encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1.5))

history = autoencoder.fit(X_train, X_train, epochs=20)

codings = encoder.predict(X_train)

import matplotlib.pyplot as plt
from Util17 import save_fig

fig = plt.figure(figsize=(4,3))
plt.plot(codings[:,0], codings[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
save_fig("linear_autoencoder_pca_plot")
plt.show()









