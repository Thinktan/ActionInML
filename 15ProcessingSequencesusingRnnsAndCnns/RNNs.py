from Util15 import generate_time_series, save_fig
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import matplotlib as mpl

np.random.seed(42)

# Generate the Dataset
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)


def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", legend=True):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bo", label="Target")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    if legend and (y or y_pred):
        plt.legend(fontsize=14, loc="upper left")

# fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
# for col in range(3):
#     plt.sca(axes[col])
#     plot_series(X_valid[col, :, 0], y_valid[col, 0],
#                 y_label=("$x(t)$" if col==0 else None),
#                legend=(col == 0))
#save_fig("time_series_plot")
# plt.show()

# Computing Some Baselines
# method 1: naive predictions (just predict the last observed value)
y_pred = X_valid[:, -1]
print("Naive predictions: %f" % (np.mean(keras.losses.mean_squared_error(y_valid, y_pred))))
# plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
# plt.show()


# method 2: linear predictions
np.random.seed(42)
tf.random.set_seed(42)

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[50, 1]),
#     keras.layers.Dense(1)
# ])

# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, y_train, epochs=20,
#                     validation_data=(X_valid, y_valid))
# print("Linear Predictions: %f" % (model.evaluate(X_valid, y_valid)))

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

# plot_learning_curves(history.history["loss"], history.history["val_loss"])
# plt.show()



# Using a Simple RNN
# model = keras.models.Sequential([
#     keras.layers.SimpleRNN(1, input_shape=[None, 1])
# ])
#
# optimizer = keras.optimizers.Adam(learning_rate=0.005)
# model.compile(loss="mse", optimizer=optimizer)
# history = model.fit(X_train, y_train, epochs=20,
#                     validation_data=(X_valid, y_valid))
#
# print("Simple RNN: %f" % (model.evaluate(X_valid, y_valid)))

# Deep RNNs
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
print("Deep RNN: %f" % (model.evaluate(X_valid, y_valid)))


# Forecasting Several Steps Ahead

