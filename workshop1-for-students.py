
import numpy as np
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.lr = lr

    def forward(self, X):
        # In forward model we feed the data X through the network.

        return

    def loss(self, y_pred, y_true):
        # MSE loss

        return

    def bptt_update(self, X, h, y_pred, y_true):
        # Using backpropagation through time (bptt) to update the weights
        # X: (batch, seq_len, input_size)

    def train(self, X, y, epochs=50, batch_size=32, verbose=True):

        return


def generate_sine_sequences(n_samples=2000, seq_len=20, input_size=1, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 50, n_samples * seq_len * input_size)
    data = np.sin(x) + 0.1 * rng.randn(n_samples * seq_len * input_size)
    X = data.reshape(n_samples, seq_len, input_size)
    rolled = np.roll(data, -1).reshape(n_samples, seq_len, input_size)
    y_last = rolled[:, -1, :]
    return X.astype(np.float32), y_last.astype(np.float32)


# Creating the data
X, y = generate_sine_sequences(n_samples=1500, seq_len=20, input_size=1)
# train/test split
split = int(0.8 * X.shape[0])
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Training the network
rnn = RNN(input_size=1, hidden_size=80, output_size=1, lr=1e-3)
losses = rnn.train(X_train, y_train, epochs=600, batch_size=8, verbose=True)

# Test set
_, y_pred_test = rnn.forward(X_test)
test_loss, _ = rnn.loss(y_pred_test, y_test)
print(f"\nTest MSE: {test_loss:.6f}\n")


# Plotting the predictions.
plt.plot(y_test)
plt.plot(y_pred_test)
plt.show()
plt.savefig("rnn_pred.png")
